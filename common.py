"""Shared setup: model loading, the demo preset, prompts, schema -> token compilation."""
import copy
import json
import os
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")

CONTEXT = (
    "Card ending 4412 made 9 purchases in 3 minutes across 4 countries. "
    "The cardholder called to say the card is still in their pocket."
)

SCHEMA = {
    "priority": {"type": "enum", "description": "Incident priority", "choices": ["HIGH", "LOW"],
                 "descriptions": {"HIGH": "needs action within the hour",
                                   "LOW": "can wait for the normal review queue"}},
    "decision": {"type": "enum", "description": "Case decision",
                 "choices": ["FRAUD", "FRAUD_REVIEW", "CLEAR"],
                 "descriptions": {"FRAUD": "confirmed fraudulent activity, block the card",
                                   "FRAUD_REVIEW": "suspicious, needs a human to look",
                                   "CLEAR": "legitimate activity, no action needed"}},
    "escalate": {"type": "boolean", "description": "Whether escalation is required"},
}


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(attn_implementation="sdpa"):
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if os.environ.get("TINY"):
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        cfg.hidden_size, cfg.intermediate_size = 128, 256
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads = 2, 4, 2
        cfg._attn_implementation = attn_implementation
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation=attn_implementation)
    return tok, model.to(device()).float().eval()


def chat(tok, system, user, prefill):
    """Model-agnostic prompt: the model's own template, then prefill the assistant turn."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    # enable_thinking=False closes the reasoning block; without it Qwen3 opens one and never emits JSON.
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + prefill


def naive_prompt(tok, context, schema):
    lines = ["{"]
    for name, f in schema.items():
        rhs = "boolean" if f["type"] == "boolean" else " | ".join(f'"{c}"' for c in f["choices"])
        lines.append(f'  "{name}": {rhs}, // {f["description"]}')
    lines.append("}")
    system = ("You are a precise data extraction system. Output ONLY a valid JSON object matching "
              "the schema below.\n\nJSON Schema:\n" + "\n".join(lines))
    return chat(tok, system, context, "{\n  ")


def parallel_prompt(tok, context, schema):
    """A field per line, plus one indented line per option that has a description.

    Our model's choice questions put each option's criteria text into the sequence
    (see ``openjev.encode.render_options``); a baseline that never sees that text is
    judged against information it was not given. A field with no ``descriptions``
    renders exactly as before - that case isn't optional, it's what the existing
    callers (``1_vanilla_decoding.py``, the tests on ``common.SCHEMA``) depend on.
    """
    lines = []
    for n, f in schema.items():
        lines.append(f'  "{n}": {f["description"]}')
        descriptions = f.get("descriptions") or {}
        for choice in choices_of(f):
            if choice in descriptions:
                lines.append(f'    {choice}: {descriptions[choice]}')
    return chat(tok, "Classify JSON attributes:\n" + "\n".join(lines), context, "{\n")


def compile_field(tok, name, field):
    if field["type"] == "boolean":
        choices = ["true", "false"]
        suffix = f"  {json.dumps(name)}: "
        conts = [c + "," for c in choices]
    else:
        choices = field["choices"]
        escaped = [json.dumps(c, ensure_ascii=False)[1:-1] for c in choices]
        shared = os.path.commonprefix(escaped)
        suffix = f'  {json.dumps(name)}: "{shared}'
        conts = [c[len(shared):] + '"' for c in escaped]
    cand_ids = [tok.encode(c, add_special_tokens=False) for c in conts]
    return SimpleNamespace(
        name=name, choices=choices, suffix=suffix, conts=conts, cand_ids=cand_ids,
        suffix_ids=tok.encode(suffix, add_special_tokens=False),
        collision=len({ids[0] for ids in cand_ids}) < len(cand_ids))


def repeat_cache(cache, n):
    branch = copy.deepcopy(cache)
    branch.batch_repeat_interleave(n)
    return branch


def choices_of(field):
    if field["type"] == "boolean":
        return ["true", "false"]
    choices = field["choices"]
    if not 2 <= len(choices) <= 255:
        raise ValueError(f"enum takes 2-255 choices, got {len(choices)}")
    if any(not isinstance(c, str) or not c for c in choices) or len(set(choices)) != len(choices):
        raise ValueError("choices must be unique nonempty strings")
    return choices


def build_trie(tok, schema):
    nodes, parents, depths, children = [], [], [], {}

    def add(parent, token):
        key = (parent, token)
        if key not in children:
            children[key] = len(nodes)
            nodes.append(token)
            parents.append(parent)
            depths.append(0 if parent == -1 else depths[parent] + 1)
        return children[key]

    score_nodes, score_tokens, score_cand = [], [], []
    counts, lengths, labels, paths = [], [], [], []
    cand = 0
    for name, field in schema.items():
        suffix = f"  {json.dumps(name)}: "
        choices = choices_of(field)
        if field["type"] == "boolean":
            conts = [c + "," for c in choices]
        else:
            suffix += '"'
            conts = [json.dumps(c, ensure_ascii=False)[1:] for c in choices]
        suffix_ids = tok.encode(suffix, add_special_tokens=False)
        node = -1
        for token in suffix_ids:
            node = add(node, token)
        counts.append(len(conts))
        for choice, cont in zip(choices, conts):
            seq = tok.encode(cont, add_special_tokens=False)
            labels.append((name, choice))
            lengths.append(len(seq))
            paths.append((suffix_ids, seq))
            cur = node
            for j, token in enumerate(seq):
                score_nodes.append(cur)
                score_tokens.append(token)
                score_cand.append(cand)
                if j < len(seq) - 1:
                    cur = add(cur, token)
            cand += 1

    kids = [[] for _ in nodes]
    roots = []
    for n, p in enumerate(parents):
        (roots if p == -1 else kids[p]).append(n)
    starts, ends, clock = [0] * len(nodes), [0] * len(nodes), 0
    stack = [(n, False) for n in reversed(roots)]
    while stack:
        n, seen = stack.pop()
        if seen:
            ends[n] = clock
        else:
            starts[n] = clock
            clock += 1
            stack.append((n, True))
            stack.extend((c, False) for c in reversed(kids[n]))

    return SimpleNamespace(nodes=nodes, depths=depths, starts=starts, ends=ends,
                           score_nodes=score_nodes, score_tokens=score_tokens,
                           score_cand=score_cand, counts=counts, lengths=lengths,
                           labels=labels, paths=paths)


def tree_scores(tok, model, context, schema, trie=None):
    """Mean token log-prob of every candidate of every field, in two forward passes."""
    trie = trie if trie is not None else build_trie(tok, schema)
    inputs = tok(parallel_prompt(tok, context, schema), add_special_tokens=False,
                 return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True)
    forward_passes = 1
    cache, P, N = out.past_key_values, inputs.input_ids.shape[1], len(trie.nodes)

    s, e = torch.tensor(trie.starts), torch.tensor(trie.ends)
    visible = (s.view(1, -1) <= s.view(-1, 1)) & (s.view(-1, 1) < e.view(1, -1))
    visible = torch.cat([torch.ones(N, P, dtype=torch.bool), visible], 1)
    mask = torch.zeros(1, 1, N, P + N, dtype=model.dtype).masked_fill(
        ~visible, torch.finfo(model.dtype).min).to(model.device)
    positions = (torch.tensor([trie.depths]) + P).to(model.device)

    out = model(input_ids=torch.tensor([trie.nodes], device=model.device), attention_mask=mask,
                position_ids=positions, past_key_values=cache, use_cache=True)
    forward_passes += 1
    logprobs = out.logits[0].float().log_softmax(-1)
    edges = logprobs[torch.tensor(trie.score_nodes, device=model.device),
                     torch.tensor(trie.score_tokens, device=model.device)]
    scores = torch.zeros(len(trie.lengths), device=model.device).scatter_add_(
        0, torch.tensor(trie.score_cand, device=model.device), edges)
    scores = scores / torch.tensor(trie.lengths, dtype=torch.float, device=model.device)
    return SimpleNamespace(scores=scores, trie=trie, prefix_len=P, visible=visible,
                           forward_passes=forward_passes)


def decide(tok, model, context, schema, temperature=1.0, trie=None):
    """Jev's surface: every question answered from one prefill, as typed choices with probabilities."""
    result = tree_scores(tok, model, context, schema, trie)
    answers, offset = {}, 0
    for (name, field), count in zip(schema.items(), result.trie.counts):
        probs = torch.softmax(result.scores[offset:offset + count] / max(temperature, 1e-4), -1)
        answers[name] = SimpleNamespace(choices=choices_of(field), probs=probs.tolist(),
                                        scores=result.scores[offset:offset + count].tolist())
        offset += count
    return SimpleNamespace(answers=answers, prefix_len=result.prefix_len,
                           nodes=len(result.trie.nodes), forward_passes=result.forward_passes)


def pack(tok, state, questions):
    """One sequence: the state, then each question with its options. Returns ids, segment ids
    (0 = state, i+1 = question i), the decision position per question, the option positions, and the
    first token of each option - the one the model would write if it answered."""
    ids = tok.encode(state, add_special_tokens=False)
    segments = [0] * len(ids)
    decisions, options, candidates = [], [], []
    for q, (instructions, choices) in enumerate(questions):
        head = tok.encode(f"\nQ: {instructions}\n", add_special_tokens=False)
        ids += head
        segments += [q + 1] * len(head)
        spans = []
        for choice in choices:
            piece = tok.encode(f"- {choice}\n", add_special_tokens=False)
            ids += piece
            segments += [q + 1] * len(piece)
            spans.append(len(ids) - 1)
        tail = tok.encode("A:", add_special_tokens=False)
        ids += tail
        segments += [q + 1] * len(tail)
        decisions.append(len(ids) - 1)
        options.append(spans)
        first = [tok.encode(f" {c}", add_special_tokens=False)[0] for c in choices]
        if len(set(first)) < len(first):
            clash = [c for c, t in zip(choices, first) if first.count(t) > 1]
            raise ValueError(
                f"{clash} share a first token, and this readout only looks at the first token - "
                "it would return an exact tie. Rename them, or score whole continuations the way "
                "extras/first_token_collisions.py does.")
        candidates.append(first)
    return ids, segments, decisions, options, candidates


def attention_mask(model, segments, bidirectional_state=False):
    """Questions see the state and themselves, never each other. With bidirectional_state the
    state block also attends forwards within itself - that is the only difference between the
    pure decoder and the hybrid."""
    length = len(segments)
    seg = torch.tensor(segments)
    causal = torch.arange(length).view(-1, 1) >= torch.arange(length).view(1, -1)
    key_is_state = seg.view(1, -1) == 0
    same_question = (seg.view(1, -1) == seg.view(-1, 1)) & ~key_is_state
    if bidirectional_state:
        visible = key_is_state | (same_question & causal)
    else:
        visible = (key_is_state | same_question) & causal
    return torch.zeros(1, 1, length, length, dtype=model.dtype).masked_fill(
        ~visible, torch.finfo(model.dtype).min).to(model.device), visible


def encode(tok, model, state, questions, bidirectional_state=False):
    """Frozen backbone, one forward. Per question: the vocabulary logits at the decision position,
    and the option token ids to read off them."""
    ids, segments, decisions, options, candidates = pack(tok, state, questions)
    mask, _ = attention_mask(model, segments, bidirectional_state)
    state_length = segments.count(0)
    positions = torch.tensor([[i if s == 0 else state_length + i - segments.index(s)
                               for i, s in enumerate(segments)]]).to(model.device)
    logits = model(input_ids=torch.tensor([ids], device=model.device), attention_mask=mask,
                   position_ids=positions).logits[0].float().cpu()
    return [(logits[d], torch.tensor(c)) for d, c in zip(decisions, candidates)]


def prefill(tok, model, state):
    """Push the state through once and keep its KV cache."""
    ids = tok.encode(state, add_special_tokens=False)
    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=True)
    return out.past_key_values, len(ids)


def question_mask(model, segments, state_length):
    """Questions see the whole cached state, themselves causally, and never each other."""
    width = len(segments)
    seg = torch.tensor(segments)
    causal = torch.arange(width).view(-1, 1) >= torch.arange(width).view(1, -1)
    same_question = seg.view(1, -1) == seg.view(-1, 1)
    visible = torch.cat([torch.ones(width, state_length, dtype=torch.bool),
                         same_question & causal], 1)
    mask = torch.zeros(1, 1, width, state_length + width, dtype=model.dtype).masked_fill(
        ~visible, torch.finfo(model.dtype).min).to(model.device)
    return mask, visible
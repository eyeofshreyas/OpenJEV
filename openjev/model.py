"""The decision model: one logit per option marker, from a bidirectional encoder.

A question and a state are encoded (see ``openjev.encode``) into one sequence with
one ``[MASK]`` marker per option. The encoder runs over the whole sequence so every
marker sees the instructions, the other options, and the state at once; a small
head refines the marker hidden states (deliberately with no positional encoding,
so it cannot learn anything from option order); and a scorer MLP turns each
marker's hidden state into exactly one logit. Softmax over a question's markers
is the answer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, ModernBertConfig, ModernBertModel

DEFAULT_BACKBONE = "answerdotai/ModernBERT-base"


def tiny_config(vocab_size: int) -> ModernBertConfig:
    """A tiny ModernBERT config for tests, so they never download pretrained weights.

    The special-token ids must be passed explicitly: the config otherwise defaults
    them to the real tokenizer's ids (50281/50282/50283), which trips
    ``nn.Embedding``'s ``padding_idx`` assertion against a small ``vocab_size``.
    """
    return ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers: int = 2):
        super().__init__()
        d = encoder.config.hidden_size
        self.encoder = encoder
        self.type_emb = nn.Embedding(3, d)
        self.head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, batch_first=True, norm_first=True),
            head_layers,
            enable_nested_tensor=False,
        )
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())

        d = hidden.size(-1)
        gathered = hidden.gather(1, marker_pos[:, :, None].expand(-1, -1, d))
        logits = self.scorer(gathered).squeeze(-1)
        logits = logits.masked_fill(~marker_mask, -1e4)
        return logits.float()


def build_model(backbone: str | None = None, config=None, head_layers: int = 2) -> DecisionModel:
    """Build a ``DecisionModel``.

    With ``config`` given, the encoder is a freshly initialized ``ModernBertModel``
    from that config (no download) — used by tests via ``tiny_config``. Otherwise
    the encoder is loaded from ``backbone`` (default ``answerdotai/ModernBERT-base``).
    """
    if config is not None:
        encoder = ModernBertModel(config)
    else:
        encoder = AutoModel.from_pretrained(backbone or DEFAULT_BACKBONE)
    return DecisionModel(encoder, head_layers=head_layers)


def probabilities(logits: torch.Tensor, marker_mask: torch.Tensor, qtype=None, temperature=None) -> torch.Tensor:
    """Softmax logits into a probability distribution over a question's real options.

    Divides by ``temperature[qtype]`` when both are given, then masks out padded
    marker slots with ``-inf`` (so they get exactly zero probability) and softmaxes.
    """
    if qtype is not None and temperature is not None:
        logits = logits / temperature[qtype][:, None]
    logits = logits.masked_fill(~marker_mask, float("-inf"))
    return torch.softmax(logits, dim=-1)

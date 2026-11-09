from openjev.encode import build_sequence, collate


def test_collate_pads_to_longest_and_masks(tok):
    a = build_sequence(tok, "short", {"type": "noul", "instructions": "i"})
    b = build_sequence(tok, "a much longer state " * 20,
                       {"type": "choice", "instructions": "i", "criteria": {"x": "", "y": "", "z": ""}})
    batch = collate([a, b], tok.pad_token_id)
    L = max(len(a.ids), len(b.ids))
    assert batch["input_ids"].shape == (2, L)
    assert batch["attention_mask"][0].sum().item() == len(a.ids)
    assert (batch["input_ids"][0, len(a.ids):] == tok.pad_token_id).all()


def test_collate_marker_mask_matches_option_counts(tok):
    a = build_sequence(tok, "s", {"type": "noul", "instructions": "i"})
    b = build_sequence(tok, "s", {"type": "choice", "instructions": "i",
                                  "criteria": {"x": "", "y": "", "z": ""}})
    batch = collate([a, b], tok.pad_token_id)
    assert batch["marker_mask"].shape == (2, 3)
    assert batch["marker_mask"][0].tolist() == [True, True, False]
    assert batch["qtype"].tolist() == [2, 0]


def test_collate_marker_pos_points_at_mask_tokens(tok):
    encs = [build_sequence(tok, "s", {"type": "noul", "instructions": "i"})]
    batch = collate(encs, tok.pad_token_id)
    pos = batch["marker_pos"][0, :2]
    assert (batch["input_ids"][0, pos] == tok.mask_token_id).all()

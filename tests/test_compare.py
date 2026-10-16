import json
import sys

import numpy as np
import pytest

from jeju_lora import compare
from jeju_lora.data import write_jsonl


def rows(preds, gold):
    pairs = enumerate(zip(gold, preds, strict=True))
    return [{"row_id": i, "gold": g, "pred": p} for i, (g, p) in pairs]


def test_identical_predictions_have_zero_diff():
    gold = [0, 1, 2, 3, 4, 5, 6] * 5
    g, pa, pb = compare.paired(rows(gold, gold), rows(gold, gold))
    out = compare.bootstrap_diff(g, pa, pb, n=50)
    assert out["diff"] == 0 and out["ci95"] == [0.0, 0.0] and out["agreement"] == 1.0


def test_better_model_has_positive_interval():
    rng = np.random.default_rng(1)
    gold = rng.integers(0, 7, 500)
    good = gold.copy()
    bad = np.where(rng.random(500) < 0.5, (gold + 1) % 7, gold)
    g, pa, pb = compare.paired(rows(good, gold), rows(bad, gold))
    out = compare.bootstrap_diff(g, pa, pb, n=200)
    assert out["f1_macro_a"] == 1.0 and out["ci95"][0] > 0 and out["share_diff_le_0"] == 0


def test_mismatched_items_rejected():
    with pytest.raises(ValueError, match="row_id"):
        compare.paired(rows([0], [0]), rows([0, 1], [0, 1]))
    with pytest.raises(ValueError, match="정답"):
        compare.paired(rows([0], [0]), rows([0], [1]))


def test_cli_writes_report(tmp_path, monkeypatch):
    gold = [0, 1, 2, 3] * 10
    write_jsonl(rows(gold, gold), tmp_path / "a.jsonl")
    write_jsonl(rows([0] * 40, gold), tmp_path / "b.jsonl")
    argv = ["compare", "a", "b", "--n", "20"]
    argv += ["--pred-dir", str(tmp_path), "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    compare.main()
    out = json.loads((tmp_path / "compare_a__b.json").read_text())
    assert out["a"] == "a" and out["diff"] > 0 and out["n_items"] == 40

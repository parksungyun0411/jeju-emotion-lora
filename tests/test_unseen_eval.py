import json
import sys

import numpy as np
import pytest

from jeju_lora import unseen_eval
from jeju_lora.data import write_jsonl


@pytest.fixture
def run(tmp_path):
    train = [{"jeju": "a", "label": 0}, {"jeju": "b", "label": 1}]
    # 앞 두 행은 train에 같은 문장이 있고, 뒤 두 행은 없다
    test = [
        {"jeju": "a", "label": 0},
        {"jeju": "b", "label": 1},
        {"jeju": "c", "label": 0},
        {"jeju": "d", "label": 1},
    ]
    write_jsonl(train, tmp_path / "train.jsonl")
    write_jsonl(test, tmp_path / "test.jsonl")
    labels = np.array([0, 1, 0, 1])
    probs = np.eye(7)[[0, 1, 0, 2]]  # 마지막 행만 틀림
    (tmp_path / "run").mkdir()
    np.savez(tmp_path / "run" / "step3_dual_probs.npz", probs=probs, labels=labels)
    return tmp_path


def test_seen_mask():
    mask = unseen_eval.seen_in_train([{"jeju": "x"}], [{"jeju": "x"}, {"jeju": "y"}])
    assert mask.tolist() == [True, False]


def test_scores_split_by_seen(run, monkeypatch):
    argv = [
        "unseen_eval",
        "--train", str(run / "train.jsonl"),
        "--test", str(run / "test.jsonl"),
        "--run-dir", str(run / "run"),
        "--out", str(run / "out" / "r"),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    unseen_eval.main()
    out = json.loads((run / "out" / "r.json").read_text())
    assert out["seen_rows"] == 2 and list(out["models"]) == ["step3_dual"]
    res = out["models"]["step3_dual"]
    assert res["all"]["accuracy"] == 0.75
    assert res["seen"]["accuracy"] == 1.0 and res["unseen"]["accuracy"] == 0.5
    assert res["unseen"]["recall"]["행복"] == {"n": 1, "recall": 0.0}
    assert res["unseen"]["recall"]["공포"] == {"n": 0, "recall": None}
    md = (run / "out" / "r.md").read_text()
    assert "| Dual-Gated KR-BERT | 0.750 | 1.000 | 0.500 |" in md


def test_label_order_mismatch(run):
    with pytest.raises(ValueError, match="순서"):
        unseen_eval.evaluate(
            [{"jeju": "a", "label": 0}],
            [{"jeju": "a", "label": 1}] * 4,
            run / "run",
        )

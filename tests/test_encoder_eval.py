import json
import sys

import numpy as np
import pytest

from jeju_lora import encoder_eval
from jeju_lora.data import read_jsonl, write_jsonl


@pytest.fixture
def setup(tmp_path):
    split_dir = tmp_path / "splits_sentence"
    test = [{"row_id": 10 + i, "label": lab} for i, lab in enumerate([0, 1, 2, 1])]
    write_jsonl(test, split_dir / "test.jsonl")
    labels = np.array([0, 1, 2, 1])
    dual = np.eye(7)[[0, 1, 2, 2]]  # 마지막 문항만 틀림
    single = np.eye(7)[[0, 1, 1, 1]]  # 세 번째 문항만 틀림
    np.savez(tmp_path / "dual.npz", probs=dual, labels=labels)
    np.savez(tmp_path / "single.npz", probs=single, labels=labels)
    write_jsonl(
        [{"row_id": 10, "human_label": 0}, {"row_id": 13, "human_label": 2}],
        tmp_path / "gold.jsonl",
    )
    return tmp_path, split_dir


def run(monkeypatch, tmp_path, split_dir, *extra):
    argv = [
        "encoder_eval",
        "--test", str(split_dir / "test.jsonl"),
        "--dual-npz", str(tmp_path / "dual.npz"),
        "--gold", str(tmp_path / "gold.jsonl"),
        "--pred-dir", str(tmp_path / "pred"),
        "--metrics-dir", str(tmp_path / "metrics"),
        *extra,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    encoder_eval.main()


def test_single_model_scores_dual_only(setup, monkeypatch):
    tmp_path, split_dir = setup
    run(monkeypatch, tmp_path, split_dir, "--tag", "dual", "--device", "mps", "--precision", "fp32")
    out = json.loads((tmp_path / "metrics" / "dual.json").read_text())
    assert out["alpha"] is None and out["split"] == "splits_sentence"
    assert out["backend"] == "encoder" and out["device"] == "mps"
    assert out["machine_label"]["accuracy"] == 0.75
    # 재라벨: row 10 → 0 맞음, row 13 → 재라벨 2, 예측 2 → 맞음
    assert out["human_label"]["n"] == 2 and out["human_label"]["accuracy"] == 1.0
    rows = read_jsonl(tmp_path / "pred" / "dual.jsonl")
    assert [r["pred"] for r in rows] == [0, 1, 2, 2] and "pred_koelectra" not in rows[0]


def test_ensemble_needs_alpha(setup, monkeypatch):
    tmp_path, split_dir = setup
    with pytest.raises(SystemExit):
        run(monkeypatch, tmp_path, split_dir, "--single-npz", str(tmp_path / "single.npz"))


def test_ensemble_combines_probs(setup, monkeypatch):
    tmp_path, split_dir = setup
    run(
        monkeypatch, tmp_path, split_dir,
        "--single-npz", str(tmp_path / "single.npz"), "--alpha", "0.6", "--tag", "ens",
    )  # fmt: skip
    rows = read_jsonl(tmp_path / "pred" / "ens.jsonl")
    # α=0.6이면 두 모델이 갈리는 문항에서 Dual을 따른다
    assert [r["pred"] for r in rows] == [0, 1, 2, 2]
    assert [r["pred_koelectra"] for r in rows] == [0, 1, 1, 1]
    out = json.loads((tmp_path / "metrics" / "ens.json").read_text())
    assert out["alpha"] == 0.6

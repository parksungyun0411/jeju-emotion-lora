"""성능 루프의 torch 없는 부분: 입력 구성, test 라벨 제거, val 선택 도구."""

import json
import tarfile

import numpy as np
import pytest
from sklearn.metrics import f1_score

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.perf import ensemble, inputs, kernel
from jeju_lora.perf.train_cls import class_weights, parse_args


def test_dedup_train_keeps_first_row_per_sentence():
    rows = [
        {"row_id": 1, "jeju": "a", "std": "A", "label": 3},
        {"row_id": 2, "jeju": "b", "std": "B", "label": 0},
        {"row_id": 3, "jeju": "a", "std": "A", "label": 3},
    ]
    assert [r["row_id"] for r in inputs.dedup_train(rows)] == [1, 2]
    assert len(inputs.select_train(rows, "dup")) == 3


def test_dedup_train_rejects_conflicting_labels():
    rows = [
        {"row_id": 1, "jeju": "a", "std": "A", "label": 3},
        {"row_id": 2, "jeju": "a", "std": "A", "label": 4},
    ]
    with pytest.raises(ValueError):
        inputs.dedup_train(rows)


def test_text_pair_puts_standard_first_by_default():
    row = {"jeju": "제주", "std": "표준"}
    assert inputs.text_pair(row, "pair") == ("표준", "제주")
    assert inputs.text_pair(row, "pair_jeju") == ("제주", "표준")
    assert inputs.text_pair(row, "jeju") == ("제주", None)
    assert inputs.text_pair(row, "std") == ("표준", None)
    with pytest.raises(ValueError):
        inputs.text_pair(row, "x")


def test_llm_text_includes_requested_fields():
    row = {"jeju": "제주", "std": "표준"}
    assert "표준어: 표준\n제주어: 제주\n감정:" in inputs.llm_text(row, "pair")
    assert "표준어" not in inputs.llm_text(row, "jeju")
    assert "제주어" not in inputs.llm_text(row, "std")


def test_strip_labels_and_normalize_ws():
    rows = [{"row_id": 1, "jeju": "a", "std": "b", "label": 2}]
    assert inputs.strip_labels(rows) == [{"row_id": 1, "jeju": "a", "std": "b"}]
    assert inputs.normalize_ws("  가  나\t다 ") == "가 나 다"


def test_dataset_test_file_has_no_labels(tmp_path, monkeypatch, items):
    split = tmp_path / "splits"
    for name in ("train", "val", "test"):
        write_jsonl(items, split / f"{name}.jsonl")
    monkeypatch.setattr(kernel, "SPLIT_DIR", split)
    manifest = kernel.build_dataset(tmp_path / "ds")
    assert set(manifest) == {"train.jsonl", "val.jsonl", "test_nolabel.jsonl", kernel.DATA_TGZ}
    with tarfile.open(tmp_path / "ds" / kernel.DATA_TGZ) as tf:
        tf.extractall(tmp_path / "x", filter="data")
    test = read_jsonl(tmp_path / "x" / "test_nolabel.jsonl")
    assert len(test) == len(items) and all("label" not in r for r in test)
    meta = json.loads((tmp_path / "ds" / "dataset-metadata.json").read_text())
    assert meta["id"] == "seongyunpark0411/jeju-perf-private"


def test_class_weights():
    labels = [0] * 70 + [1] * 30 + [2, 3, 4, 5, 6] * 10
    assert class_weights(labels, "none") is None
    w = class_weights(labels, "balanced")
    assert w[6] > w[1] > w[0] and np.isclose(w.mean(), 1.0)
    assert class_weights(labels, "sqrt")[6] < w[6]


def test_parse_args_defaults():
    a = parse_args(["--name", "x", "--model", "m", "--data-dir", "d", "--out-dir", "o"])
    assert (a.arch, a.input, a.train_mode, a.fp16) == ("seqcls", "pair", "dedup", True)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 500)
    p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 7, 500))
    ref = f1_score(y, p, labels=list(range(7)), average="macro", zero_division=0)
    assert ensemble.macro_f1(y, p) == pytest.approx(ref)


def _noisy_logits(n=700, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(7, n, p=[0.3, 0.17, 0.25, 0.12, 0.07, 0.04, 0.05])
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y] += 1.0
    logits[:, 0] += 0.8  # 다수 클래스 쪽으로 치우친 모델
    return y, ensemble.log_softmax(logits)


def test_fit_bias_does_not_lower_in_sample_f1():
    y, lp = _noisy_logits()
    b = ensemble.fit_bias(lp, y)
    assert ensemble.macro_f1(y, (lp + b).argmax(-1)) >= ensemble.macro_f1(y, lp.argmax(-1))
    assert 0 <= ensemble.oof_bias_f1(lp, y) <= 1


def test_folds_partition_and_stratify():
    y, _ = _noisy_logits()
    fs = ensemble.folds(y, 5)
    allidx = np.concatenate(fs)
    assert sorted(allidx.tolist()) == list(range(len(y)))
    for f in fs:
        assert abs((y[f] == 0).mean() - (y == 0).mean()) < 0.05


def test_greedy_ensemble_weights_sum_to_one():
    y, a = _noisy_logits(seed=1)
    _, b = _noisy_logits(seed=2)
    w, bias = ensemble.greedy_ensemble([a, b], y)
    assert w.sum() == pytest.approx(1.0) and bias.shape == (7,)


def test_apply_recipe_combines_members_without_labels(tmp_path):
    from jeju_lora.perf.final import apply_recipe

    rng = np.random.default_rng(0)
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        np.savez(d / "logits.npz", test_logits=rng.normal(size=(2, 9, 7)), test_row_id=np.arange(9))
    recipe = {
        "members": [
            {"dir": str(tmp_path / "a"), "eval_index": 1, "weight": 0.5},
            {"dir": str(tmp_path / "b"), "eval_index": 0, "weight": 0.5},
        ],
        "bias": [0.0] * 6 + [100.0],
    }
    ids, pred = apply_recipe(recipe)
    assert ids.tolist() == list(range(9)) and set(pred.tolist()) == {6}

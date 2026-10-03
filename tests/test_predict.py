import json

import numpy as np
import pytest

from jeju_lora.data import write_jsonl
from jeju_lora.encoder_eval import align
from jeju_lora.generation import OpenAIChatBackend
from jeju_lora.labels import EMOTION_LABELS, LABEL_NAMES
from jeju_lora.predict import load_gold, run_predictions, score


class StubBackend:
    """정답 라벨을 그대로 말하되, row_id가 3의 배수면 엉뚱한 말을 하는 가짜 모델."""

    def __init__(self, items):
        self.answers = {it["jeju"]: it for it in items}
        self.calls = 0

    def generate(self, batch):
        self.calls += 1
        out = []
        for messages in batch:
            jeju = messages[-1]["content"].split("\n")[0].removeprefix("제주어: ")
            it = self.answers[jeju]
            out.append("잘 모르겠어요" if it["row_id"] % 3 == 0 else EMOTION_LABELS[it["label"]])
        return out


def test_predictions_and_scores(items, tmp_path):
    backend = StubBackend(items)
    preds = run_predictions(backend, items, "jeju", [])
    assert backend.calls == 1 and len(preds) == len(items)
    invalid = [p for p in preds if p["pred"] == -1]
    assert len(invalid) == sum(it["row_id"] % 3 == 0 for it in items)

    gold_path = tmp_path / "gold.jsonl"
    write_jsonl([{"row_id": 1, "human_label": 1}, {"row_id": 2, "human_label": 0}], gold_path)
    gold = load_gold(gold_path)
    res = score(preds, gold)
    assert res["machine_label"]["accuracy"] == pytest.approx(1 - len(invalid) / len(items))
    # 사람 라벨: row 1 → 정답(슬픔), row 2 → 모델은 행복, 사람은 중립 → 오답
    assert res["human_label"]["n"] == 2
    assert res["human_label"]["accuracy"] == 0.5


def test_gold_missing_means_no_human_scores(items, tmp_path):
    assert load_gold(tmp_path / "none.jsonl") == {}
    preds = run_predictions(StubBackend(items), items[:7], "jeju", [])
    assert score(preds, {})["human_label"] is None


def test_openai_payload_constrains_to_labels():
    b = OpenAIChatBackend("http://x/v1/", "m")
    body = b.payload([{"role": "user", "content": "hi"}])
    assert b.url == "http://x/v1/chat/completions"
    assert body["structured_outputs"] == {"choice": LABEL_NAMES}
    assert body["temperature"] == 0.0
    assert "structured_outputs" not in OpenAIChatBackend(
        "http://x", "m", constrained=False
    ).payload([])
    json.dumps(body, ensure_ascii=False)


def test_encoder_npz_alignment():
    test = [{"row_id": i, "label": lab} for i, lab in enumerate([0, 1, 2])]
    probs = np.eye(7)[[0, 2, 2]]
    assert align(test, probs, np.array([0, 1, 2])) == [0, 2, 2]
    with pytest.raises(ValueError, match="분할"):
        align(test, probs, np.array([0, 2, 1]))


def test_rescore_adds_human_scores_later(items, tmp_path):
    from jeju_lora.predict import rescore_all

    preds = run_predictions(StubBackend(items), items[:14], "jeju", [])
    (tmp_path / "p").mkdir()
    (tmp_path / "m").mkdir()
    write_jsonl(preds, tmp_path / "p" / "x.jsonl")
    (tmp_path / "m" / "x.json").write_text(json.dumps({"tag": "x", **score(preds, {})}))
    (tmp_path / "m" / "orphan.json").write_text("{}")
    updated = rescore_all(tmp_path / "p", tmp_path / "m", {1: 1, 4: 4})
    assert updated == ["x"]
    out = json.loads((tmp_path / "m" / "x.json").read_text())
    assert out["tag"] == "x" and out["human_label"]["n"] == 2

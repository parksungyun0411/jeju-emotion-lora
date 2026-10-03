from collections import Counter
from datetime import datetime, timezone

import pytest

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.review.label import make_record, parse_input, pending, run
from jeju_lora.review.report import agreement, build_gold, latest
from jeju_lora.review.sample import disagreement_map, stratified_sample


def test_sample_stratified_by_emotion(items):
    s = stratified_sample(items, 70)
    assert Counter(x["machine_label"] for x in s) == {c: 10 for c in range(7)}
    assert len({x["row_id"] for x in s}) == 70
    assert s == stratified_sample(items, 70)
    assert not any(x["disagreement"] for x in s)


def test_sample_uneven_n(items):
    s = stratified_sample(items, 72)
    counts = Counter(x["machine_label"] for x in s)
    assert sum(counts.values()) == 72 and max(counts.values()) - min(counts.values()) == 1


def test_sample_prefers_disagreement(items, tmp_path):
    # 클래스마다 앞의 5개 문항은 모델이 다른 라벨을 냈다고 둔다
    rows = []
    for it in items:
        wrong = (it["label"] + 1) % 7 if it["row_id"] < 35 else it["label"]
        rows.append({"row_id": it["row_id"], "pred": wrong, "pred_dual": it["label"]})
    write_jsonl(rows, tmp_path / "p.jsonl")
    preds = disagreement_map([tmp_path / "p.jsonl"])
    s = stratified_sample(items, 70, preds, disagree_frac=0.4)
    per_class_dis = Counter(x["machine_label"] for x in s if x["disagreement"])
    assert per_class_dis == {c: 4 for c in range(7)}  # round(10 × 0.4)
    s = stratified_sample(items, 70, preds, disagree_frac=1.0)
    per_class_dis = Counter(x["machine_label"] for x in s if x["disagreement"])
    assert per_class_dis == {c: 5 for c in range(7)}  # 가진 만큼만
    assert len(s) == 70


def test_sample_rejects_bad_args(items):
    with pytest.raises(ValueError):
        stratified_sample(items, 3)
    with pytest.raises(ValueError):
        stratified_sample(items, 70, disagree_frac=1.5)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3", ("label", 3, "")),
        ("0 표준어 번역이 어색", ("label", 0, "표준어 번역이 어색")),
        ("a 반어법", ("ambiguous", None, "반어법")),
        ("s", ("skip", None, "")),
        ("q", ("quit", None, "")),
        ("7", ("invalid", None, "")),
        ("", ("invalid", None, "")),
        ("분노", ("invalid", None, "")),
    ],
)
def test_parse_input(text, expected):
    assert parse_input(text) == expected


def _sample_items():
    return [
        {
            "row_id": i,
            "jeju": f"j{i}",
            "std": f"s{i}",
            "machine_label": i % 7,
            "disagreement": False,
        }
        for i in range(5)
    ]


def test_label_session_records_and_resumes(tmp_path):
    items = _sample_items()
    out = tmp_path / "decisions.jsonl"
    answers = iter(["9", "2", "a 애매", "s", "q"])
    shown: list[str] = []
    n = run(items, out, "박성윤", False, ask=lambda _: next(answers), say=shown.append)
    assert n == 2
    recs = read_jsonl(out)
    assert [r["decision"] for r in recs] == ["label", "ambiguous"]
    assert recs[0]["human_label"] == 2 and recs[0]["reviewer"] == "박성윤"
    assert recs[0]["machine_label_shown"] is False
    assert datetime.fromisoformat(recs[0]["decided_at"]).tzinfo is not None
    assert not any("기계 라벨" in line for line in shown)  # 블라인드
    # 이어서 하면 건너뛴 문항(row 2)부터 다시 나온다
    assert [it["row_id"] for it in pending(items, recs, "박성윤")] == [2, 3, 4]
    assert len(pending(items, recs, "다른검수자")) == 5


def _rec(row_id, reviewer, label, minute, decision="label"):
    item = {"row_id": row_id}
    now = datetime(2026, 10, 20, 10, minute, tzinfo=timezone.utc)
    return make_record(item, reviewer, decision, label, "", False, now=now)


def test_report_latest_majority_ties_ambiguous():
    decisions = [
        _rec(0, "A", 1, 0),
        _rec(0, "A", 0, 5),  # 고쳐 판정 → 마지막(0)을 쓴다
        _rec(0, "B", 0, 1),
        _rec(1, "A", 3, 0),
        _rec(1, "B", 4, 1),  # 동률 → gold에서 제외
        _rec(2, "A", None, 0, "ambiguous"),
        _rec(3, "A", 3, 0),
    ]
    final = latest(decisions)
    gold = {g["row_id"]: g for g in build_gold(final)}
    assert gold[0]["human_label"] == 0 and gold[0]["n_reviewers"] == 2
    assert 1 not in gold and 2 not in gold
    assert gold[3]["human_label"] == 3

    machine = {0: 0, 1: 3, 2: 2, 3: 3}
    rep = agreement(final, machine)
    a = rep["reviewers"]["A"]
    assert a["decided"] == 4 and a["ambiguous"] == 1 and a["labelled"] == 3
    assert a["agree_with_gpt4o"] == 1.0
    pair = rep["inter_reviewer"]["A|B"]
    # 공유 문항 0(0,0), 1(3,4): po=1/2, pe = (½·½)+(¼·0)+(0·¼)... = 0.25 → κ=(0.5-0.25)/0.75
    assert pair["shared"] == 2
    assert pair["kappa"] == pytest.approx((0.5 - 0.25) / 0.75)

import json

from jeju_lora.v2.collect import agreement, check_batch
from jeju_lora.v2.pool import clean_dialect, clean_standard, cue_group, split_of


def test_pair_notation_keeps_dialect_side():
    assert clean_dialect("겅허고 예초 벌초(하다그네)/(하다가)") == "겅허고 예초 벌초하다그네"
    assert clean_dialect("아 예초하다가(이)/(#이) 뱀 나온거야") == "아 예초하다가이 뱀 나온거야"


def test_tags_and_anonymization():
    assert clean_dialect("귀찮아? {laughing}") == "귀찮아?"
    assert clean_dialect("얘 이거 어디 &company-name& 아니면") == "얘 이거 어디 OO 아니면"
    assert clean_standard("&name1& 왔어 {laughing}") == "OO 왔어"


def test_unclear_speech_is_dropped():
    assert clean_dialect("일일이 산소 못하(())") is None
    assert clean_dialect("((이제여행))다녀") is None
    assert clean_dialect("이 한* 놈 튀믄") is None
    assert clean_dialect("나도 겅/()) 해") is None


def test_split_is_deterministic_and_by_file():
    assert split_of("DZES20000002") == split_of("DZES20000002")
    splits = {split_of(f"DZES2{i:07d}") for i in range(2000)}
    assert splits == {"train", "val", "test"}


def test_cue_group():
    assert cue_group("그거 진짜 무서웠어") == "공포"
    assert cue_group("내일 학교 가") is None


def test_agreement_perfect_and_counts():
    a = {str(i): {"label": i % 7, "conf": "high"} for i in range(14)}
    r = agreement(a, a)
    assert r["n"] == 14 and r["agree"] == 1.0 and abs(r["macro_f1_mean"] - 1.0) < 1e-9
    assert r["agree_when_both_conf"]["high"]["n"] == 14


def _write(path, rows):
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def test_check_batch_parts_and_head(tmp_path):
    items = [{"id": f"f.{i}", "jeju": f"{i:03d}번째 문장입니다"} for i in range(60)]
    _write(tmp_path / "001.jsonl", items)
    labels = tmp_path / "labels"
    labels.mkdir()
    rows = [{"id": r["id"], "head": r["jeju"][:8], "label": 0, "conf": "high"} for r in items]
    _write(labels / "001.p1.jsonl", rows[:50])
    assert check_batch(tmp_path / "001.jsonl", labels) == ["missing 001.p2.jsonl"]
    _write(labels / "001.p2.jsonl", rows[50:])
    assert check_batch(tmp_path / "001.jsonl", labels) == []
    shifted = rows[:50] + [dict(r, head=rows[i - 1]["head"]) for i, r in enumerate(rows[50:], 50)]
    _write(labels / "001.p2.jsonl", shifted[50:])
    assert check_batch(tmp_path / "001.jsonl", labels) == ["head mismatch: 10"]


def test_finalize_majority_rules(tmp_path):
    from jeju_lora.v2.collect import finalize

    data = tmp_path / "v2"
    (data / "items").mkdir(parents=True)
    lab = data / "labels"
    for d in ("train", "evalA", "evalB", "evalC"):
        (lab / d).mkdir(parents=True)
    item = lambda i: {"id": i, "jeju": f"{i} 제주", "std": f"{i} 표준"}  # noqa: E731
    _write(data / "items" / "train.jsonl", [item("t1"), item("t2")])
    _write(data / "items" / "val.jsonl", [item("v1"), item("v2"), item("v3"), item("v4")])
    _write(data / "items" / "test.jsonl", [])
    _write(lab / "train" / "001.p1.jsonl", [{"id": "t1", "label": 2, "conf": "high"}])
    _write(
        lab / "evalA" / "001.p1.jsonl",
        [
            {"id": v, "label": x, "conf": "mid"}
            for v, x in [("v1", 1), ("v2", 1), ("v3", 1), ("v4", 1)]
        ],
    )
    _write(
        lab / "evalB" / "001.p1.jsonl",
        [
            {"id": v, "label": x, "conf": "mid"}
            for v, x in [("v1", 1), ("v2", 3), ("v3", 3), ("v4", 3)]
        ],
    )
    _write(
        lab / "evalC" / "001.p1.jsonl",
        [{"id": "v2", "label": 3, "conf": "mid"}, {"id": "v3", "label": 5, "conf": "mid"}],
    )
    stats = finalize(data, lab)
    assert stats["train"]["kept"] == 1 and stats["train"]["unlabeled"] == 1
    assert stats["val"] == {
        "kept": 2,
        "agreed": 1,
        "majority": 1,
        "no_majority": 1,
        "pending_adjudication": 1,
        "dist": {"행복": 1, "분노": 1},
    }
    exported = [
        json.loads(x) for x in (tmp_path / "splits_v2" / "val.jsonl").read_text().splitlines()
    ]
    assert [(r["row_id"], r["id"], r["label"]) for r in exported] == [(0, "v1", 1), (1, "v2", 3)]

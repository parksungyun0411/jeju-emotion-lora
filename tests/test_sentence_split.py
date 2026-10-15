import pandas as pd

from jeju_lora.data import JEJU_COL, LABEL_COL, STD_COL, prepare_frame, stratified_split_712
from jeju_lora.leakage import duplicate_profile, render_markdown, split_overlap
from jeju_lora.sentence_split import sentence_split, split_summary


def dup_frame() -> pd.DataFrame:
    """고유 문장 7 × 30개, 소수 클래스는 여러 번 복제. 라벨이 엇갈린 문장 1개."""
    rows = []
    for i in range(7 * 30):
        copies = 1 + (i % 7)  # 라벨이 클수록 많이 복제
        for _ in range(copies):
            rows.append({JEJU_COL: f"문장 {i}", STD_COL: f"표준 {i}", LABEL_COL: i % 7})
    rows.append({JEJU_COL: "문장 0", STD_COL: "표준 0", LABEL_COL: 3})  # 문장 0에 다른 라벨
    return pd.DataFrame(rows)


def test_row_split_leaks_duplicates():
    df = prepare_frame(dup_frame())
    tr, _, te = stratified_split_712(df)
    o = split_overlap(tr, te)
    assert o["in_train"] > 0
    assert 0 < o["share"] <= 1


def test_sentence_split_has_no_overlap_and_drops_conflicts():
    df = prepare_frame(dup_frame())
    s = sentence_split(df)
    tr, va, te = set(s.train[JEJU_COL]), set(s.val[JEJU_COL]), set(s.test[JEJU_COL])
    assert not (tr & va or tr & te or va & te)
    assert "문장 0" not in tr | va | te
    # val·test는 문장당 한 행, train은 복제 행을 유지
    assert s.test[JEJU_COL].is_unique and s.val[JEJU_COL].is_unique
    assert len(s.train) > s.train[JEJU_COL].nunique()
    summary = split_summary(s)
    assert summary["test"]["overlap_with_train"] == 0
    assert summary["val"]["overlap_with_train"] == 0
    assert summary["val_test_overlap"] == 0
    # 고유 문장 기준 7:1:2 (라벨 엇갈린 1문장 제외 209개)
    n = len(tr | va | te)
    assert n == 209
    assert abs(len(te) - 0.2 * n) <= 2


def test_sentence_split_is_deterministic():
    df = prepare_frame(dup_frame())
    a = sentence_split(df).test["row_id"].tolist()
    b = sentence_split(df.sample(frac=1, random_state=1)).test["row_id"].tolist()
    assert a == b


def test_duplicate_profile_and_markdown():
    df = prepare_frame(dup_frame())
    d = duplicate_profile(df)
    assert d["unique_sentences"] == 210
    assert d["conflicting_sentences"] == 1
    assert d["duplicate_rows"] == len(df) - 210
    tr, _, te = stratified_split_712(df)
    md = render_markdown(
        {
            "source_file": "x.xlsx",
            "duplicates": d,
            "original_split_overlap": split_overlap(tr, te),
            "sentence_split": split_summary(sentence_split(df)),
        }
    )
    assert "문장 단위 분할" in md and "중립" in md
    assert "문장 1" not in md  # 문장 원문은 보고서에 나오지 않는다


def test_sentence_split_for_original_returns_original_columns(tmp_path):
    from jeju_lora.run_original import ORIGINAL_COLS, sentence_split_for_original

    path = tmp_path / "data.csv"
    dup_frame().to_csv(path, index=False)
    split_fn = sentence_split_for_original(path)
    train, val, test = split_fn(None, seed=42)
    for part in (train, val, test):
        assert list(part.columns) == ["row_id", *ORIGINAL_COLS]
    assert not set(train["제주어 문장"]) & set(test["제주어 문장"])
    # 원 코드가 넘기는 df와 무관하게 같은 분할을 돌려준다
    assert split_fn(dup_frame())[2]["row_id"].tolist() == test["row_id"].tolist()

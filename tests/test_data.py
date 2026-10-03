from pathlib import Path

import pytest

from jeju_lora.data import (
    LABEL_COL,
    load_splits,
    prepare_frame,
    split_manifest,
    stratified_split_712,
    to_records,
)
from jeju_lora.verify_split import (
    DEFAULT_ORIGINAL_REPO,
    ORIGINAL_FILE,
    compare,
    load_original_split_fn,
)

ORIGINAL_PRESENT = (DEFAULT_ORIGINAL_REPO / ORIGINAL_FILE).exists()


def test_prepare_drops_na_and_keeps_original_row_ids(raw_df):
    df = prepare_frame(raw_df)
    assert len(df) == len(raw_df) - 3
    assert {5, 50, 100}.isdisjoint(df["row_id"])
    # row_id는 dropna 이전 행 번호다
    row = df[df["row_id"] == 6].iloc[0]
    assert row["제주어 문장"] == raw_df.iloc[6]["제주어 문장"]
    assert df[LABEL_COL].dtype.kind == "i"


def test_prepare_requires_columns(raw_df):
    with pytest.raises(ValueError, match="필수 열"):
        prepare_frame(raw_df.drop(columns=["표준어 문장"]))


def test_split_ratio_disjoint_stratified(raw_df):
    df = prepare_frame(raw_df)
    tr, va, te = stratified_split_712(df)
    n = len(df)
    assert len(te) == pytest.approx(0.2 * n, abs=1)
    assert len(va) == pytest.approx(0.1 * n, abs=2)
    ids = [set(x["row_id"]) for x in (tr, va, te)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == set(df["row_id"])
    counts = te[LABEL_COL].value_counts()
    assert counts.max() - counts.min() <= 1


def test_split_is_deterministic(raw_df):
    df = prepare_frame(raw_df)
    a = stratified_split_712(df)[2]["row_id"].tolist()
    b = stratified_split_712(df.copy())[2]["row_id"].tolist()
    assert a == b
    c = stratified_split_712(df, seed=7)[2]["row_id"].tolist()
    assert a != c


def test_extra_column_does_not_change_selection(raw_df):
    """row_id 열을 붙여도 원 코드(열 없음)와 같은 행이 test로 간다."""
    df = prepare_frame(raw_df)
    with_id = stratified_split_712(df)[2]
    without_id = stratified_split_712(df.drop(columns=["row_id"]))[2]
    assert with_id["제주어 문장"].tolist() == without_id["제주어 문장"].tolist()


@pytest.mark.skipif(not ORIGINAL_PRESENT, reason="원 저장소(jeju-emotion-analysis)가 디스크에 없다")
def test_matches_original_split_function(raw_df):
    original = load_original_split_fn()
    result = compare(prepare_frame(raw_df), original)
    assert all(v["match"] for v in result.values())


def test_compare_detects_different_split(raw_df):
    def shifted(df, seed=42):
        return stratified_split_712(df, seed=seed + 1)

    result = compare(prepare_frame(raw_df), shifted)
    assert not result["test"]["match"]


def test_load_splits_from_xlsx_and_manifest(raw_df, tmp_path: Path):
    path = tmp_path / "merged.xlsx"
    raw_df.to_excel(path, index=False)
    splits = load_splits(path)
    recs = to_records(splits.test)
    assert set(recs[0]) == {"row_id", "jeju", "std", "label"}
    m = split_manifest(splits, path)
    assert m["sizes"]["test"] == len(splits.test)
    assert len(m["test_row_ids_sha256"]) == 64
    assert sum(m["label_counts_test"].values()) == len(splits.test)


def test_sizes_match_original_report_for_same_row_count():
    """원 보고서의 분할 크기(89,118/12,732/25,463)는 dropna 후 127,313행일 때 나온다."""
    import pandas as pd

    from jeju_lora.verify_split import ORIGINAL_SIZES

    n = sum(ORIGINAL_SIZES.values())
    df = pd.DataFrame({"row_id": range(n), LABEL_COL: [i % 7 for i in range(n)]})
    tr, va, te = stratified_split_712(df)
    assert {"train": len(tr), "val": len(va), "test": len(te)} == ORIGINAL_SIZES

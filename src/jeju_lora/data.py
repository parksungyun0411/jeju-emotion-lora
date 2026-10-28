"""원 저장소 재현 코드와 같은 데이터·분할·시드(발표 때의 시드는 남아 있지 않다).

원 저장소 재현 코드(jeju-emotion-analysis/src/jeju_kr_bert.py)의 처리 순서를 그대로 따른다.
  1. pd.read_excel(merged_jeju_balanced_sorted.xlsx)
  2. dropna(['제주어 문장', '표준어 문장', '감정번호']).reset_index(drop=True)
  3. stratified_split_712(df, seed=42): test 20% → 남은 것의 1/8을 val

원 코드는 행 ID를 남기지 않는다. 여기서는 1단계 직후의 행 번호를 `row_id`로 붙여 둔다.
열이 하나 늘어도 train_test_split의 섞는 순서는 행 수·라벨·시드로만 정해지므로
선택되는 행은 원 코드와 같다(tests/test_data.py에서 원 함수와 직접 대조).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

JEJU_COL = "제주어 문장"
STD_COL = "표준어 문장"
LABEL_COL = "감정번호"
SEED = 42
DEFAULT_XLSX = "data/raw/merged_jeju_balanced_sorted.xlsx"
SPLIT_DIR = "data/splits"


def stratified_split_712(df: pd.DataFrame, label_col: str = LABEL_COL, seed: int = SEED):
    """원 저장소 재현 코드의 stratified_split_712와 같은 호출 순서·인자."""
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[label_col])
    train, val = train_test_split(
        train_val, test_size=1 / 8, random_state=seed, stratify=train_val[label_col]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"지원하지 않는 형식: {path.suffix}")


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """원 코드의 전처리(dropna·형 변환)를 적용하고 row_id를 붙인다."""
    missing = {JEJU_COL, STD_COL, LABEL_COL} - set(raw.columns)
    if missing:
        raise ValueError(f"필수 열 없음: {sorted(missing)}")
    df = raw.copy()
    df.insert(0, "row_id", range(len(df)))
    df = df.dropna(subset=[JEJU_COL, STD_COL, LABEL_COL]).reset_index(drop=True)
    df[JEJU_COL] = df[JEJU_COL].astype(str)
    df[STD_COL] = df[STD_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_splits(path: str | Path = DEFAULT_XLSX, seed: int = SEED) -> Splits:
    df = prepare_frame(read_table(path))
    return Splits(*stratified_split_712(df, seed=seed))


def to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {"row_id": int(r.row_id), "jeju": r.jeju, "std": r.std, "label": int(r.label)}
        for r in df.rename(columns={JEJU_COL: "jeju", STD_COL: "std", LABEL_COL: "label"})
        .loc[:, ["row_id", "jeju", "std", "label"]]
        .itertuples(index=False)
    ]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ids_digest(row_ids) -> str:
    """순서를 포함한 row_id 목록의 sha256. 분할 일치 여부를 텍스트 없이 기록할 때 쓴다."""
    return hashlib.sha256(",".join(str(int(i)) for i in row_ids).encode()).hexdigest()


def split_manifest(splits: Splits, source: str | Path) -> dict:
    import sklearn

    return {
        "source_file": Path(source).name,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "sizes": {k: len(getattr(splits, k)) for k in ("train", "val", "test")},
        "label_counts_test": {
            str(k): int(v) for k, v in splits.test[LABEL_COL].value_counts().sort_index().items()
        },
        "test_row_ids_sha256": ids_digest(splits.test["row_id"]),
        "val_row_ids_sha256": ids_digest(splits.val["row_id"]),
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="원 저장소 재현 코드와 같은 7:1:2 분할을 jsonl로 저장")
    p.add_argument("--data", default=DEFAULT_XLSX)
    p.add_argument("--out-dir", default=SPLIT_DIR)
    p.add_argument("--manifest", default="reports/split_manifest.json")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"{args.data} 없음. 원 프로젝트의 라벨링 결과 파일을 이 경로에 두어야 한다 "
            "(docs/runpod_setup.md '데이터' 절)."
        )
    splits = load_splits(args.data)
    for name in ("train", "val", "test"):
        write_jsonl(to_records(getattr(splits, name)), Path(args.out_dir) / f"{name}.jsonl")
    manifest = split_manifest(splits, args.data)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["sizes"]))


if __name__ == "__main__":
    main()

"""문장 단위 분할: 같은 제주어 문장이 두 분할에 걸치지 않게 나눈다.

원 분할(행 단위 7:1:2)은 test 행의 70.4%가 train에 같은 문장으로 들어 있다
(고유 문장 기준 63.0%, reports/leakage.md).
여기서는 고유 문장을 단위로 라벨 층화 7:1:2를 하고, 학습 방식은 원 프로젝트와 같게 두려고
train에는 그 문장의 복제 행을 모두 넣는다(원 파일의 클래스 균형용 복제를 그대로 유지).
val·test는 문장당 한 행만 둔다. 같은 문장에 라벨이 둘 이상 붙은 문장은 정답을 정할 수 없어
모든 분할에서 뺀다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from jeju_lora.data import (
    DEFAULT_XLSX,
    JEJU_COL,
    LABEL_COL,
    SEED,
    Splits,
    ids_digest,
    prepare_frame,
    read_table,
    to_records,
    write_jsonl,
)

SENTENCE_SPLIT_DIR = "data/splits_sentence"


def sentence_split(df: pd.DataFrame, seed: int = SEED) -> Splits:
    """prepare_frame 결과를 받아 문장 단위로 나눈다. 대표 행은 문장의 가장 작은 row_id."""
    n_labels = df.groupby(JEJU_COL)[LABEL_COL].nunique()
    conflict = set(n_labels[n_labels > 1].index)
    clean = df[~df[JEJU_COL].isin(conflict)]
    sents = (
        clean.sort_values("row_id")
        .groupby(JEJU_COL, sort=False)
        .agg(row_id=("row_id", "first"), label=(LABEL_COL, "first"))
        .reset_index()
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    train_val, test = train_test_split(
        sents, test_size=0.2, random_state=seed, stratify=sents["label"]
    )
    train, val = train_test_split(
        train_val, test_size=1 / 8, random_state=seed, stratify=train_val["label"]
    )
    train_rows = clean[clean[JEJU_COL].isin(set(train[JEJU_COL]))].sort_values("row_id")
    by_id = clean.set_index("row_id", drop=False)
    val_rows = by_id.loc[val["row_id"]].reset_index(drop=True)
    test_rows = by_id.loc[test["row_id"]].reset_index(drop=True)
    return Splits(train_rows.reset_index(drop=True), val_rows, test_rows)


def split_summary(s: Splits) -> dict:
    seen = set(s.train[JEJU_COL])
    out = {}
    for name in ("train", "val", "test"):
        part = getattr(s, name)
        out[name] = {
            "rows": int(len(part)),
            "unique_sentences": int(part[JEJU_COL].nunique()),
        }
        if name != "train":
            out[name]["overlap_with_train"] = int(part[JEJU_COL].isin(seen).sum())
    out["val_test_overlap"] = int(s.test[JEJU_COL].isin(set(s.val[JEJU_COL])).sum())
    out["test_label_counts"] = {
        str(k): int(v) for k, v in s.test[LABEL_COL].value_counts().sort_index().items()
    }
    out["test_row_ids_sha256"] = ids_digest(s.test["row_id"])
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="문장 단위 7:1:2 분할을 jsonl로 저장")
    p.add_argument("--data", default=DEFAULT_XLSX)
    p.add_argument("--out-dir", default=SENTENCE_SPLIT_DIR)
    p.add_argument("--manifest", default="reports/split_manifest_sentence.json")
    args = p.parse_args()

    s = sentence_split(prepare_frame(read_table(args.data)))
    for name in ("train", "val", "test"):
        write_jsonl(to_records(getattr(s, name)), Path(args.out_dir) / f"{name}.jsonl")
    summary = split_summary(s)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k in ("train", "val", "test")}))


if __name__ == "__main__":
    main()

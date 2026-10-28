"""원 분할의 중복 문장 누수 진단.

원 라벨링 파일은 클래스 균형을 맞추면서 같은 제주어 문장을 여러 번 넣었다. 행 단위로 무작위
분할하면 같은 문장이 train과 test에 함께 들어간다. 이 모듈은 모델 없이 데이터만으로
그 규모를 센다. 문장 원문은 출력하지 않는다(AI Hub 재배포 제한).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jeju_lora.data import (
    DEFAULT_XLSX,
    JEJU_COL,
    LABEL_COL,
    prepare_frame,
    read_table,
    stratified_split_712,
)
from jeju_lora.labels import EMOTION_LABELS as LABELS
from jeju_lora.verify_split import ORIGINAL_SIZES


def duplicate_profile(df: pd.DataFrame) -> dict:
    """문장 중복 규모와 같은 문장에 다른 라벨이 붙은 경우."""
    vc = df[JEJU_COL].value_counts()
    n_labels = df.groupby(JEJU_COL)[LABEL_COL].nunique()
    conflict = n_labels[n_labels > 1].index
    per_label = {}
    for lab, g in df.groupby(LABEL_COL):
        per_label[int(lab)] = {
            "rows": int(len(g)),
            "unique_sentences": int(g[JEJU_COL].nunique()),
            "max_copies": int(g[JEJU_COL].value_counts().max()),
        }
    return {
        "rows": int(len(df)),
        "unique_sentences": int(vc.size),
        "duplicate_rows": int(len(df) - vc.size),
        "max_copies": int(vc.max()),
        "conflicting_sentences": int(len(conflict)),
        "conflicting_rows": int(df[JEJU_COL].isin(conflict).sum()),
        "per_label": per_label,
    }


def split_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """test 행 가운데 같은 문장이 train에 있는 비율(전체, 라벨별)."""
    seen = set(train[JEJU_COL])
    in_train = test[JEJU_COL].isin(seen)
    pairs = set(zip(train[JEJU_COL], train[LABEL_COL], strict=True))
    same_pair = pd.Series(
        [(s, y) in pairs for s, y in zip(test[JEJU_COL], test[LABEL_COL], strict=True)],
        index=test.index,
    )
    per_label = {}
    for lab, idx in test.groupby(LABEL_COL).groups.items():
        per_label[int(lab)] = {
            "test_rows": int(len(idx)),
            "in_train": int(in_train.loc[idx].sum()),
            "share": round(float(in_train.loc[idx].mean()), 4),
            "unseen_rows": int((~in_train.loc[idx]).sum()),
        }
    return {
        "test_rows": int(len(test)),
        "in_train": int(in_train.sum()),
        "share": round(float(in_train.mean()), 4),
        "same_sentence_and_label": int(same_pair.sum()),
        "unseen_rows": int((~in_train).sum()),
        "unique_sentences": int(test[JEJU_COL].nunique()),
        "in_train_unique_sentences": int(test.loc[in_train, JEJU_COL].nunique()),
        "unseen_unique_sentences": int(test.loc[~in_train, JEJU_COL].nunique()),
        "per_label": per_label,
    }


def md_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_markdown(report: dict) -> str:
    d, o = report["duplicates"], report["original_split_overlap"]
    lines = [
        "# 원 분할의 중복 문장 누수 진단",
        "",
        "`make leakage`로 생성. 모델 없이 데이터와 분할만으로 계산한 값이다. "
        f"원본: `{report['source_file']}`, 전처리 후 {d['rows']:,}행.",
        "",
        "## 문장 중복",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 행 | {d['rows']:,} |",
        f"| 서로 다른 제주어 문장 | {d['unique_sentences']:,} |",
        f"| 중복 행 (행 − 고유 문장) | {d['duplicate_rows']:,} |",
        f"| 한 문장의 최대 복제 수 | {d['max_copies']} |",
        "| 같은 문장에 라벨이 둘 이상인 문장 / 행 | "
        f"{d['conflicting_sentences']:,} / {d['conflicting_rows']:,} |",
        "",
        "| 감정 | 행 | 고유 문장 | 최대 복제 |",
        "|---|---:|---:|---:|",
    ]
    for lab, v in sorted(d["per_label"].items()):
        lines.append(
            md_row(
                LABELS[int(lab)], f"{v['rows']:,}", f"{v['unique_sentences']:,}", v["max_copies"]
            )
        )
    lines += [
        "",
        "## 원 분할(행 단위 7:1:2, seed 42)의 train·test 겹침",
        "",
        "원 코드와 같은 절차·시드로 이 파일을 다시 나눈 분할이다. 원 보고서의 분할 크기("
        + " / ".join(f"{v:,}" for v in ORIGINAL_SIZES.values())
        + f")는 전처리 후 {sum(ORIGINAL_SIZES.values()):,}행 기준이고 "
        f"이 파일은 {d['rows']:,}행이라, 원 실험의 분할과는 행 수부터 다르다.",
        "",
        f"test {o['test_rows']:,}행 가운데 **{o['in_train']:,}행({o['share']:.1%})** 은 "
        "같은 제주어 문장이 train에 있다. "
        f"문장과 라벨이 모두 같은 행은 {o['same_sentence_and_label']:,}행이다. "
        f"train에 없는 문장은 {o['unseen_rows']:,}행"
        f"(고유 문장 {o['unseen_unique_sentences']:,}개)이다. "
        f"행이 아니라 고유 문장으로 세면 test {o['unique_sentences']:,}문장 가운데 "
        f"{o['in_train_unique_sentences']:,}문장"
        f"({o['in_train_unique_sentences'] / o['unique_sentences']:.1%})이 train에 있다.",
        "",
        "| 감정 | test 행 | train에 같은 문장 | 비율 | train에 없는 행 |",
        "|---|---:|---:|---:|---:|",
    ]
    for lab, v in sorted(o["per_label"].items()):
        lines.append(
            md_row(
                LABELS[int(lab)],
                f"{v['test_rows']:,}",
                f"{v['in_train']:,}",
                f"{v['share']:.1%}",
                f"{v['unseen_rows']:,}",
            )
        )
    s = report.get("sentence_split")
    if s:
        lines += [
            "",
            "## 문장 단위 분할",
            "",
            "고유 문장을 라벨 층화 7:1:2로 나누고(seed 42), "
            "train에는 그 문장의 복제 행을 모두 넣고 val·test는 문장당 한 행만 둔다. "
            "라벨이 둘 이상인 문장은 모든 분할에서 뺐다.",
            "",
            "| 분할 | 행 | 고유 문장 | train과 겹치는 문장 |",
            "|---|---:|---:|---:|",
        ]
        for name in ("train", "val", "test"):
            v = s[name]
            overlap = v.get("overlap_with_train", "—")
            lines.append(md_row(name, f"{v['rows']:,}", f"{v['unique_sentences']:,}", overlap))
        lines += ["", "| 감정 | test 문장 |", "|---|---:|"]
        for lab, n in sorted(s["test_label_counts"].items()):
            lines.append(f"| {LABELS[int(lab)]} | {n:,} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    from jeju_lora.sentence_split import sentence_split, split_summary

    p = argparse.ArgumentParser(description="원 분할의 중복 문장 누수 진단")
    p.add_argument("--data", default=DEFAULT_XLSX)
    p.add_argument("--out", default="reports/leakage")
    args = p.parse_args()

    df = prepare_frame(read_table(args.data))
    tr, _, te = stratified_split_712(df)
    report = {
        "source_file": Path(args.data).name,
        "duplicates": duplicate_profile(df),
        "original_split_overlap": split_overlap(tr, te),
        "sentence_split": split_summary(sentence_split(df)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    out.with_suffix(".md").write_text(render_markdown(report))
    o = report["original_split_overlap"]
    print(
        f"원 분할 test {o['test_rows']:,}행 중 "
        f"train에 같은 문장 {o['in_train']:,}행 ({o['share']:.1%})"
    )


if __name__ == "__main__":
    main()

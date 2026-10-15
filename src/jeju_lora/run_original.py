"""원 프로젝트 학습 코드를 그대로 돌리되, 분할만 바꿔 끼운다.

원 저장소(jeju-emotion-analysis/src/jeju_kr_bert.py)는 읽기만 한다. 모듈을 경로로 불러와
`stratified_split_712`만 이 저장소의 분할 함수로 바꾼 뒤 원래의 단계 함수를 호출한다.
학습 코드·하이퍼파라미터·앙상블 방식은 원 프로젝트와 같다.

  --split row       원 분할(행 단위 7:1:2). 원 보고서 수치 재현용
  --split sentence  문장 단위 분할(sentence_split.py). 같은 문장이 train과 test에 걸치지 않는다

  python -m jeju_lora.run_original --split sentence --steps dual koelectra ensemble \
      --data data/raw/merged_jeju_balanced_sorted.xlsx --out runs/original_sentence
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from jeju_lora.data import prepare_frame, read_table
from jeju_lora.sentence_split import sentence_split
from jeju_lora.verify_split import DEFAULT_ORIGINAL_REPO, ORIGINAL_FILE

ORIGINAL_COLS = ["제주어 문장", "표준어 문장", "감정번호"]


def load_original_module(repo: str | Path):
    path = Path(repo) / ORIGINAL_FILE
    spec = importlib.util.spec_from_file_location("jeju_kr_bert_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sentence_split_for_original(data_path: str | Path):
    """원 코드의 stratified_split_712 자리에 넣을 함수. 원 코드가 넘기는 df는 쓰지 않고
    같은 파일을 이 저장소 방식(row_id 유지)으로 다시 읽어 문장 단위로 나눈다."""
    s = sentence_split(prepare_frame(read_table(data_path)))
    frames = tuple(
        part[["row_id", *ORIGINAL_COLS]].reset_index(drop=True) for part in (s.train, s.val, s.test)
    )

    def split_fn(df, label_col="감정번호", seed=42):  # noqa: ARG001 — 원 함수와 같은 시그니처
        return tuple(f.copy() for f in frames)

    return split_fn


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["row", "sentence"], required=True)
    p.add_argument("--steps", nargs="+", default=["dual", "koelectra", "ensemble"])
    p.add_argument("--data", default="data/raw/merged_jeju_balanced_sorted.xlsx")
    p.add_argument("--original-repo", default=str(DEFAULT_ORIGINAL_REPO))
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=None, help="비우면 원 기본값")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--amp", action="store_true", help="원 코드의 --fp16(bf16 autocast)")
    args = p.parse_args()

    orig = load_original_module(args.original_repo)
    if args.split == "sentence":
        orig.stratified_split_712 = sentence_split_for_original(args.data)
    if args.amp:
        import torch

        orig.AMP_DTYPE = torch.bfloat16

    hp = dict(orig.DEFAULT_HP)
    if args.epochs:
        hp["epochs"] = args.epochs
    if args.batch_size:
        hp["batch_size"] = args.batch_size
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps(
            {"split": args.split, "steps": args.steps, "hp": hp, "amp": args.amp},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    data = str(Path(args.data).resolve())
    if "dual" in args.steps:
        orig.run_step_dual("step3_dual", data, "snunlp/KR-BERT-char16424", hp, str(out))
    if "koelectra" in args.steps:
        orig.run_step_single(
            "branch1_koelectra",
            data,
            "제주어 문장",
            "monologg/koelectra-base-v3-discriminator",
            hp,
            str(out),
        )
    if "ensemble" in args.steps:
        orig.run_step_ensemble(
            "step7_ensemble",
            str(out / "step3_dual_probs.npz"),
            str(out / "branch1_koelectra_probs.npz"),
            str(out),
        )


if __name__ == "__main__":
    main()

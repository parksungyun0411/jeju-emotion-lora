"""원 프로젝트 앙상블의 test 예측(npz)을 이 저장소의 row_id에 붙여 다시 채점한다.

원 코드(jeju_kr_bert.py save_report)는 test 분할 순서대로 probs·labels를 npz에 저장한다.
분할이 같으면 data/splits/test.jsonl과 순서가 같으므로 라벨 배열이 일치하는지로 확인한다.
이렇게 하면 앙상블을 다시 돌리지 않고도 재라벨 문항에서 F1을 잴 수 있다.

  python -m jeju_lora.encoder_eval \
      --dual-npz runs/original/step3_dual_probs.npz \
      --single-npz runs/original/branch1_koelectra_probs.npz --alpha 0.5

모델 하나만 채점하려면 --single-npz를 빼고, 기록용 --tag·--device·--precision을 준다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.predict import load_gold, score, split_name


def align(test: list[dict], probs: np.ndarray, labels: np.ndarray) -> list[int]:
    """npz와 test.jsonl의 순서가 같은지 확인하고 argmax 예측을 돌려준다."""
    gold = np.array([int(t["label"]) for t in test])
    if len(gold) != len(labels) or not np.array_equal(gold, labels):
        raise ValueError("npz 라벨 순서가 test.jsonl과 다르다 — 분할이 다르다")
    return probs.argmax(-1).astype(int).tolist()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", default="data/splits/test.jsonl")
    p.add_argument("--dual-npz", required=True)
    p.add_argument("--single-npz", default=None, help="비우면 --dual-npz 하나만 채점한다")
    p.add_argument("--alpha", type=float, default=None, help="원 실행에서 고른 α (앙상블일 때)")
    p.add_argument("--device", default=None, help="기록용 학습 장치 (예: mps, Tesla T4)")
    p.add_argument("--precision", default=None, help="기록용 학습 정밀도 (예: bf16-autocast, fp32)")
    p.add_argument("--gold", default="reviews/gold.jsonl")
    p.add_argument("--tag", default="encoder_ensemble")
    p.add_argument("--pred-dir", default="outputs/predictions")
    p.add_argument("--metrics-dir", default="reports/metrics")
    args = p.parse_args()

    test = read_jsonl(args.test)
    d = np.load(args.dual_npz)
    if args.single_npz is None:
        preds = align(test, d["probs"], d["labels"])
        rows = [
            {"row_id": t["row_id"], "gold": int(t["label"]), "pred": pr}
            for t, pr in zip(test, preds, strict=True)
        ]
    else:
        if args.alpha is None:
            raise SystemExit("앙상블에는 --alpha가 필요하다")
        s = np.load(args.single_npz)
        probs = args.alpha * d["probs"] + (1 - args.alpha) * s["probs"]
        preds_ens = align(test, probs, d["labels"])
        preds_dual = align(test, d["probs"], d["labels"])
        preds_single = align(test, s["probs"], s["labels"])
        rows = [
            {
                "row_id": t["row_id"],
                "gold": int(t["label"]),
                "pred": pe,
                "pred_dual": pd_,
                "pred_koelectra": ps,
            }
            for t, pe, pd_, ps in zip(test, preds_ens, preds_dual, preds_single, strict=True)
        ]
    write_jsonl(rows, Path(args.pred_dir) / f"{args.tag}.jsonl")
    out = {
        "tag": args.tag,
        "backend": "encoder",
        "alpha": args.alpha,
        "split": split_name(args.test),
        "device": args.device,
        "precision": args.precision,
        **score(rows, load_gold(args.gold)),
    }
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.metrics_dir) / f"{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"{args.tag}: f1_macro={out['machine_label']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

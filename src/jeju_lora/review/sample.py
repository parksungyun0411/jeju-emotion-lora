"""검수(재라벨)용 표본 추출: 감정별 층화 + 모델 불일치 문항 우선.

불일치 = 기계 라벨(GPT-4o)과 주어진 모델 예측들이 하나라도 다른 문항.
예측 파일이 없으면 감정별 층화만 한다. test 분할에서만 뽑는다.

  python -m jeju_lora.review.sample --n 280 \
      --predictions outputs/predictions/encoder_ensemble.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.labels import NUM_LABELS

PRED_KEYS = ("pred", "pred_dual", "pred_koelectra")


def disagreement_map(pred_files: list[str | Path]) -> dict[int, set[int]]:
    """row_id → 모든 예측 파일에 나온 예측값 집합."""
    out: dict[int, set[int]] = {}
    for path in pred_files:
        for r in read_jsonl(path):
            s = out.setdefault(int(r["row_id"]), set())
            s.update(int(r[k]) for k in PRED_KEYS if k in r)
    return out


def stratified_sample(
    items: list[dict],
    n: int,
    preds: dict[int, set[int]] | None = None,
    disagree_frac: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    if n < NUM_LABELS:
        raise ValueError(f"n은 {NUM_LABELS} 이상")
    if not 0 <= disagree_frac <= 1:
        raise ValueError("disagree_frac는 [0, 1]")
    rng = random.Random(seed)
    preds = preds or {}
    per_class = [n // NUM_LABELS + (1 if c < n % NUM_LABELS else 0) for c in range(NUM_LABELS)]

    picked: list[dict] = []
    for c in range(NUM_LABELS):
        pool = [it for it in items if int(it["label"]) == c]
        dis = [it for it in pool if preds.get(it["row_id"], set()) - {c}]
        agr = [it for it in pool if not preds.get(it["row_id"], set()) - {c}]
        rng.shuffle(dis)
        rng.shuffle(agr)
        want_dis = min(len(dis), round(per_class[c] * disagree_frac)) if preds else 0
        chosen = dis[:want_dis] + agr[: per_class[c] - want_dis]
        if len(chosen) < per_class[c]:  # 일치 문항이 모자라면 불일치에서 채운다
            chosen += dis[want_dis : want_dis + per_class[c] - len(chosen)]
        for it in chosen:
            picked.append(
                {
                    "row_id": it["row_id"],
                    "jeju": it["jeju"],
                    "std": it["std"],
                    "machine_label": int(it["label"]),
                    "disagreement": bool(preds.get(it["row_id"], set()) - {c}),
                }
            )
    rng.shuffle(picked)  # 감정별로 몰려 나오지 않게 섞는다
    return picked


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/splits/test.jsonl")
    p.add_argument("--n", type=int, default=280)
    p.add_argument("--predictions", nargs="*", default=[])
    p.add_argument("--disagree-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/review/sample.jsonl")
    p.add_argument("--ids-out", default="reviews/sample_ids.json")
    args = p.parse_args()

    if not 200 <= args.n <= 300:
        print(f"주의: 계획한 규모(200~300건) 밖이다: {args.n}")
    preds = disagreement_map(args.predictions) if args.predictions else None
    sample = stratified_sample(read_jsonl(args.data), args.n, preds, args.disagree_frac, args.seed)
    write_jsonl(sample, args.out)
    Path(args.ids_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ids_out).write_text(
        json.dumps(
            {
                "n": len(sample),
                "seed": args.seed,
                "disagree_frac": args.disagree_frac,
                "prediction_files": [Path(x).name for x in args.predictions],
                "row_ids": [s["row_id"] for s in sample],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    n_dis = sum(s["disagreement"] for s in sample)
    print(f"{len(sample)}건 추출 (불일치 {n_dis}건) → {args.out}")


if __name__ == "__main__":
    main()

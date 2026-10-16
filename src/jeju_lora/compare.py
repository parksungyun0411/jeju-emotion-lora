"""두 구성의 test 예측을 같은 문항끼리 묶어 macro-F1 차이를 부트스트랩으로 잰다.

test 문항을 복원 추출로 다시 뽑아(같은 표본을 두 구성에 함께 쓴다) 차이의 분포를 만든다.
95% 구간이 0을 포함하면 이 test 크기로는 두 구성의 우열을 가리지 않는다.

  python -m jeju_lora.compare lora_kanana-1.5-2.1b_fp16 encoder_ensemble
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from jeju_lora.data import read_jsonl
from jeju_lora.labels import NUM_LABELS

CLASSES = list(range(NUM_LABELS))


def paired(a: list[dict], b: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """row_id로 두 예측을 맞춰 (정답, a 예측, b 예측)을 돌려준다."""
    da = {r["row_id"]: r for r in a}
    db = {r["row_id"]: r for r in b}
    if set(da) != set(db):
        raise ValueError("두 예측의 row_id가 다르다 — 같은 test가 아니다")
    ids = sorted(da)
    gold = np.array([da[i]["gold"] for i in ids])
    if not np.array_equal(gold, [db[i]["gold"] for i in ids]):
        raise ValueError("같은 row_id의 정답 라벨이 다르다")
    return gold, np.array([da[i]["pred"] for i in ids]), np.array([db[i]["pred"] for i in ids])


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    # 재표본마다 나오는 클래스가 달라지지 않게 7개 클래스를 고정한다. 파싱 실패(-1)는 오답이다
    return float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0))


def bootstrap_diff(
    gold: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 1000, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        i = rng.integers(0, len(gold), len(gold))
        diffs[k] = macro_f1(gold[i], pa[i]) - macro_f1(gold[i], pb[i])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_items": int(len(gold)),
        "f1_macro_a": macro_f1(gold, pa),
        "f1_macro_b": macro_f1(gold, pb),
        "diff": macro_f1(gold, pa) - macro_f1(gold, pb),
        "ci95": [float(lo), float(hi)],
        "share_diff_le_0": float(np.mean(diffs <= 0)),
        "agreement": float(np.mean(pa == pb)),
        "n_bootstrap": n,
        "seed": seed,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="예측 태그 (outputs/predictions/{a}.jsonl)")
    p.add_argument("b")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pred-dir", default="outputs/predictions")
    p.add_argument("--out-dir", default="reports/diagnostics")
    args = p.parse_args()

    gold, pa, pb = paired(
        read_jsonl(Path(args.pred_dir) / f"{args.a}.jsonl"),
        read_jsonl(Path(args.pred_dir) / f"{args.b}.jsonl"),
    )
    out = {"a": args.a, "b": args.b, **bootstrap_diff(gold, pa, pb, args.n, args.seed)}
    path = Path(args.out_dir) / f"compare_{args.a}__{args.b}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    lo, hi = out["ci95"]
    print(f"{args.a} - {args.b}: {out['diff']:+.4f} (95% 구간 {lo:+.4f} ~ {hi:+.4f})")


if __name__ == "__main__":
    main()

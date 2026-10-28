"""성능 루프 결과 고르기: 실행별 val 점수, 클래스 편향 보정, 앙상블 가중치. val만 읽는다.

모든 선택(평가 시점·편향·가중치)은 val로만 한다. 같은 val로 고르고 재면 낙관적이므로
val을 5겹으로 나눠 4겹에서 고르고 남은 1겹을 재는 교차 추정(oof)을 함께 적는다.
test 로짓은 이 모듈에서 읽지 않는다(최종 1회 평가는 final.py).

  python -m jeju_lora.perf.ensemble runs/perf/r1/* --out reports/perf_loop/r1_val.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jeju_lora.labels import LABEL_NAMES, NUM_LABELS

BIAS_GRID = np.round(np.arange(-1.5, 1.5001, 0.05), 3)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int = NUM_LABELS) -> float:
    """sklearn f1_score(average='macro', labels=0..k-1)와 같은 값을 빠르게 낸다."""
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def per_class_f1(y: np.ndarray, pred: np.ndarray, k: int = NUM_LABELS) -> dict:
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return {LABEL_NAMES[i]: round(float(v), 4) for i, v in enumerate(f1)}


def fit_bias(logp: np.ndarray, y: np.ndarray, rounds: int = 3) -> np.ndarray:
    """클래스별 로그 확률에 더할 편향을 좌표 상승으로 찾는다(macro-F1 최대)."""
    b = np.zeros(logp.shape[1])
    best = macro_f1(y, (logp + b).argmax(-1))
    for _ in range(rounds):
        improved = False
        for c in range(logp.shape[1]):
            for v in BIAS_GRID:
                trial = b.copy()
                trial[c] = v
                f = macro_f1(y, (logp + trial).argmax(-1))
                if f > best + 1e-9:
                    best, b, improved = f, trial, True
        if not improved:
            break
    return b


def folds(y: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """라벨 층화 k겹 (val 인덱스)."""
    rng = np.random.default_rng(seed)
    out = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        for j, part in enumerate(np.array_split(idx, k)):
            out[j].extend(part.tolist())
    return [np.array(sorted(f)) for f in out]


def oof_bias_f1(logp: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> float:
    """편향을 4겹에서 맞추고 남은 겹에 적용한 예측을 모아 잰 macro-F1."""
    pred = np.empty(len(y), dtype=int)
    for f in folds(y, k, seed):
        mask = np.ones(len(y), bool)
        mask[f] = False
        b = fit_bias(logp[mask], y[mask])
        pred[f] = (logp[f] + b).argmax(-1)
    return macro_f1(y, pred)


def greedy_ensemble(
    members: list[np.ndarray], y: np.ndarray, steps: int = 25, bias: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Caruana식 전진 선택(복원). 가중치(선택 횟수 비율)와 편향을 돌려준다."""
    counts = np.zeros(len(members))
    acc = np.zeros_like(members[0])
    best_f = -1.0
    for _ in range(steps):
        scores = []
        for m in members:
            cand = (acc + m) / (counts.sum() + 1)
            scores.append(macro_f1(y, cand.argmax(-1)))
        j = int(np.argmax(scores))
        if scores[j] <= best_f and counts.sum() > 0:
            break
        best_f = scores[j]
        counts[j] += 1
        acc = acc + members[j]
    w = counts / counts.sum()
    mix = sum(wi * m for wi, m in zip(w, members, strict=True))
    b = fit_bias(mix, y) if bias else np.zeros(mix.shape[1])
    return w, b


def oof_ensemble_f1(members: list[np.ndarray], y: np.ndarray, k: int = 5, seed: int = 0) -> float:
    pred = np.empty(len(y), dtype=int)
    for f in folds(y, k, seed):
        mask = np.ones(len(y), bool)
        mask[f] = False
        w, b = greedy_ensemble([m[mask] for m in members], y[mask])
        mix = sum(wi * m[f] for wi, m in zip(w, members, strict=True))
        pred[f] = (mix + b).argmax(-1)
    return macro_f1(y, pred)


def load_run(path: str | Path) -> dict | None:
    path = Path(path)
    if not (path / "logits.npz").exists():
        return None
    z = np.load(path / "logits.npz")
    meta = json.loads((path / "metrics.json").read_text())
    return {
        "name": meta["name"],
        "dir": str(path),
        "meta": meta,
        "val_logits": z["val_logits"],
        "val_label": z["val_label"],
        "val_row_id": z["val_row_id"],
    }


def run_logp(run: dict, how: str) -> np.ndarray:
    """평가 시점 고르기: best(val 최고) · last · avgK(마지막 K개 log-softmax 평균)."""
    lp = log_softmax(run["val_logits"])
    if how == "best":
        return lp[int(run["meta"]["best_index"])]
    if how == "last":
        return lp[-1]
    if how.startswith("avg"):
        return lp[-int(how[3:]) :].mean(0)
    raise ValueError(how)


def summarize(runs: list[dict]) -> dict:
    y = runs[0]["val_label"]
    for r in runs[1:]:
        if not np.array_equal(r["val_row_id"], runs[0]["val_row_id"]):
            raise ValueError(f"{r['name']}의 val 순서가 다르다")
    rows = {}
    for r in runs:
        lp = run_logp(r, "best")
        pred = lp.argmax(-1)
        b = fit_bias(lp, y)
        rows[r["name"]] = {
            "val_f1_macro": round(macro_f1(y, pred), 4),
            "best_epoch": r["meta"]["best"]["epoch"],
            "val_f1_bias_insample": round(macro_f1(y, (lp + b).argmax(-1)), 4),
            "val_f1_bias_oof": round(oof_bias_f1(lp, y), 4),
            "val_f1_per_class": per_class_f1(y, pred),
            "elapsed_min": r["meta"]["elapsed_min"],
        }
    out = {"n_val": int(len(y)), "runs": rows}
    if len(runs) > 1:
        members = [run_logp(r, "best") for r in runs]
        w, b = greedy_ensemble(members, y)
        mix = sum(wi * m for wi, m in zip(w, members, strict=True))
        pred = (mix + b).argmax(-1)
        out["ensemble"] = {
            "weights": {r["name"]: round(float(wi), 4) for r, wi in zip(runs, w, strict=True)},
            "bias": [round(float(v), 3) for v in b],
            "val_f1_insample": round(macro_f1(y, pred), 4),
            "val_f1_oof": round(oof_ensemble_f1(members, y), 4),
            "val_f1_per_class": per_class_f1(y, pred),
            # final.py가 test에 그대로 적용할 처방 (val에서만 정한 값)
            "recipe": {
                "members": [
                    {
                        "dir": r["dir"],
                        "eval_index": int(r["meta"]["best_index"]),
                        "weight": round(float(wi), 6),
                    }
                    for r, wi in zip(runs, w, strict=True)
                    if wi > 0
                ],
                "bias": [float(v) for v in b],
            },
        }
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="train_cls 출력 폴더들")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = [r for r in (load_run(x) for x in args.runs) if r is not None]
    out = summarize(runs)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

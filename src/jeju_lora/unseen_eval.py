"""원 행 단위 분할의 인코더 예측을 'train에 같은 문장이 있는 test 행'과 '없는 행'으로 나눠 채점한다.

원 행 단위 분할은 test 행의 70.4%가 train에 같은 제주어 문장을 두고 있다(reports/leakage.md).
원 프로젝트의 0.84가 문장을 외운 결과인지 보려면 두 부분을 따로 재야 한다.
train에 없는 행은 감정 분포가 크게 다르다(공포 0행, 혐오 9행, 놀람 49행).
그래서 macro-F1은 쓰지 않고 정확도와 감정별 재현율을 두 부분에서 함께 적는다.
같은 감정 안에서 비교하면 분포 차이를 피할 수 있다.
문장 원문은 출력하지 않는다(AI Hub 재배포 제한).

  python -m jeju_lora.unseen_eval --run-dir runs/original_row
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jeju_lora.data import read_jsonl
from jeju_lora.labels import LABEL_NAMES, NUM_LABELS

# 원 코드가 run 폴더에 남기는 test 확률 파일 → 표에 쓰는 이름
MODELS = {
    "step3_dual": "Dual-Gated KR-BERT",
    "branch1_koelectra": "KoELECTRA",
    "step7_ensemble": "앙상블",
}
SUBSETS = ("all", "seen", "unseen")
SUBSET_NAMES = {"all": "전체", "seen": "train에 있는 문장", "unseen": "train에 없는 문장"}


def seen_in_train(train: list[dict], test: list[dict]) -> np.ndarray:
    """test 행마다 train에 같은 제주어 문장이 있으면 True."""
    seen = {r["jeju"] for r in train}
    return np.array([r["jeju"] in seen for r in test], dtype=bool)


def subset_scores(gold: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """정확도와 감정별 재현율. 정답 행이 없는 감정은 재현율을 null로 둔다."""
    g, p = gold[mask], pred[mask]
    recall = {}
    for c in range(NUM_LABELS):
        n = int((g == c).sum())
        recall[LABEL_NAMES[c]] = {
            "n": n,
            "recall": float((p[g == c] == c).mean()) if n else None,
        }
    return {
        "n": int(mask.sum()),
        "accuracy": float((g == p).mean()) if mask.any() else None,
        "recall": recall,
    }


def evaluate(train: list[dict], test: list[dict], run_dir: Path) -> dict:
    seen = seen_in_train(train, test)
    gold = np.array([int(r["label"]) for r in test])
    masks = {"all": np.ones_like(seen), "seen": seen, "unseen": ~seen}
    out = {"test_rows": len(test), "seen_rows": int(seen.sum()), "models": {}}
    for key in MODELS:
        f = run_dir / f"{key}_probs.npz"
        if not f.exists():
            continue
        d = np.load(f)
        if not np.array_equal(d["labels"], gold):
            raise ValueError(f"{f}: npz 라벨 순서가 test.jsonl과 다르다")
        pred = d["probs"].argmax(-1)
        out["models"][key] = {s: subset_scores(gold, pred, masks[s]) for s in SUBSETS}
    return out


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def render_markdown(report: dict) -> str:
    unseen = report["test_rows"] - report["seen_rows"]
    lines = [
        "# 원 행 단위 분할: train에 있는 문장과 없는 문장",
        "",
        "`make unseen-eval`이 원 코드의 test 확률(runs/original_row/*_probs.npz)로 생성한다. "
        "손으로 고치지 않는다.",
        "",
        f"test {report['test_rows']:,}행 가운데 train에 같은 문장이 있는 행은 "
        f"{report['seen_rows']:,}행, 없는 행은 {unseen:,}행이다. "
        "없는 행에는 공포·혐오·놀람이 거의 없어 macro-F1 대신 정확도와 감정별 재현율을 적는다.",
        "",
        "| 모델 | " + " | ".join(f"정확도 ({SUBSET_NAMES[s]})" for s in SUBSETS) + " |",
        "|---|" + "---:|" * len(SUBSETS),
    ]
    for key, res in report["models"].items():
        lines.append(
            f"| {MODELS[key]} | " + " | ".join(_pct(res[s]["accuracy"]) for s in SUBSETS) + " |"
        )
    for key, res in report["models"].items():
        lines += [
            "",
            f"## {MODELS[key]} 감정별 재현율",
            "",
            "| 감정 | train에 있는 문장 (행) | 재현율 | train에 없는 문장 (행) | 재현율 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in LABEL_NAMES:
            s, u = res["seen"]["recall"][name], res["unseen"]["recall"][name]
            lines.append(
                f"| {name} | {s['n']:,} | {_pct(s['recall'])} | {u['n']:,} | {_pct(u['recall'])} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="data/splits/train.jsonl")
    p.add_argument("--test", default="data/splits/test.jsonl")
    p.add_argument("--run-dir", default="runs/original_row")
    p.add_argument("--out", default="reports/diagnostics/row_split_seen_unseen")
    args = p.parse_args()

    report = evaluate(read_jsonl(args.train), read_jsonl(args.test), Path(args.run_dir))
    if not report["models"]:
        raise SystemExit(f"{args.run_dir}에 *_probs.npz가 없다")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    for key, res in report["models"].items():
        print(f"{key}: seen {res['seen']['accuracy']:.4f} / unseen {res['unseen']['accuracy']:.4f}")


if __name__ == "__main__":
    main()

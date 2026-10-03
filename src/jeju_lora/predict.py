"""평가셋에 대해 생성 → 파싱 → 지표. 기계 라벨(GPT-4o)과 사람 검수 라벨 양쪽으로 잰다.

예:
  python -m jeju_lora.predict --backend openai --model qwen3-4b-lora-merged \
      --tag lora_qwen3-4b_bf16 --shots 0
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.labels import parse_label
from jeju_lora.metrics import classification_metrics
from jeju_lora.prompts import build_messages, pick_few_shot

DEFAULT_TEST = "data/splits/test.jsonl"
DEFAULT_TRAIN = "data/splits/train.jsonl"
DEFAULT_GOLD = "reviews/gold.jsonl"


def run_predictions(backend, items: list[dict], input_mode: str, shots: list[dict]) -> list[dict]:
    batch = [build_messages(it, input_mode, shots) for it in items]
    raw = backend.generate(batch)
    return [
        {"row_id": it["row_id"], "gold": int(it["label"]), "pred": parse_label(r), "raw": r}
        for it, r in zip(items, raw, strict=True)
    ]


def load_gold(path: str | Path) -> dict[int, int]:
    """사람 검수 라벨 {row_id: label}. 파일이 없거나 비어 있으면 빈 dict."""
    path = Path(path)
    if not path.exists():
        return {}
    return {int(r["row_id"]): int(r["human_label"]) for r in read_jsonl(path)}


def score(preds: list[dict], gold: dict[int, int]) -> dict:
    result = {
        "machine_label": classification_metrics(
            [p["gold"] for p in preds], [p["pred"] for p in preds]
        )
    }
    human = [(gold[p["row_id"]], p["pred"]) for p in preds if p["row_id"] in gold]
    result["human_label"] = (
        classification_metrics([h for h, _ in human], [p for _, p in human]) if human else None
    )
    return result


def rescore_all(pred_dir: str | Path, metrics_dir: str | Path, gold: dict[int, int]) -> list[str]:
    """저장된 예측으로 지표를 다시 계산한다. 사람 검수가 GPU 실행보다 늦게 끝나도 된다."""
    updated = []
    for mpath in sorted(Path(metrics_dir).glob("*.json")):
        ppath = Path(pred_dir) / f"{mpath.stem}.jsonl"
        if not ppath.exists():
            continue
        out = json.loads(mpath.read_text(encoding="utf-8"))
        out.update(score(read_jsonl(ppath), gold))
        mpath.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        updated.append(mpath.stem)
    return updated


def rescore_main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=rescore_all.__doc__)
    p.add_argument("--gold", default=DEFAULT_GOLD)
    p.add_argument("--pred-dir", default="outputs/predictions")
    p.add_argument("--metrics-dir", default="reports/metrics")
    args = p.parse_args()
    print(rescore_all(args.pred_dir, args.metrics_dir, load_gold(args.gold)))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["openai", "hf"], default="openai")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="openai: 서빙 이름 / hf: 경로 또는 HF ID")
    p.add_argument("--revision", default=None)
    p.add_argument("--adapter", default=None, help="hf 백엔드에서 LoRA 어댑터 경로")
    p.add_argument("--no-constrained", action="store_true", help="choice 제약 없이 자유 생성")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--data", default=DEFAULT_TEST)
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--gold", default=DEFAULT_GOLD)
    p.add_argument("--only-gold", action="store_true", help="사람 검수 문항만 평가")
    p.add_argument("--shots", type=int, default=0, help="클래스당 퓨샷 예시 수")
    p.add_argument("--input-mode", default="jeju")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--pred-dir", default="outputs/predictions")
    p.add_argument("--metrics-dir", default="reports/metrics")
    args = p.parse_args()

    items = read_jsonl(args.data)
    gold = load_gold(args.gold)
    if args.only_gold:
        if not gold:
            raise SystemExit(f"{args.gold}가 없거나 비어 있다 — 사람 검수가 먼저다")
        items = [it for it in items if it["row_id"] in gold]
    if args.limit:
        items = items[: args.limit]
    shots = pick_few_shot(read_jsonl(args.train), args.shots) if args.shots else []

    if args.backend == "openai":
        from jeju_lora.generation import OpenAIChatBackend

        backend = OpenAIChatBackend(
            args.base_url, args.model, not args.no_constrained, concurrency=args.concurrency
        )
    else:
        from jeju_lora.generation import HFBackend

        backend = HFBackend(args.model, adapter=args.adapter, revision=args.revision)

    t0 = time.perf_counter()
    preds = run_predictions(backend, items, args.input_mode, shots)
    elapsed = time.perf_counter() - t0

    write_jsonl(preds, Path(args.pred_dir) / f"{args.tag}.jsonl")
    out = {
        "tag": args.tag,
        "backend": args.backend,
        "model": args.model,
        "adapter": args.adapter,
        "shots_per_class": args.shots,
        "input_mode": args.input_mode,
        "constrained": args.backend == "openai" and not args.no_constrained,
        "data": args.data,
        "elapsed_s": elapsed,
        **score(preds, gold),
    }
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.metrics_dir) / f"{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    m = out["machine_label"]
    print(f"{args.tag}: n={m['n']} f1_macro={m['f1_macro']:.4f} invalid={m['invalid_rate']:.4f}")


if __name__ == "__main__":
    main()

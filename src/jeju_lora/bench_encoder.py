"""KoELECTRA + Dual KR-BERT 앙상블의 GPU 처리량 측정 (건당 비용 비교용).

앙상블 1건 = KoELECTRA 1회(제주어) + KR-BERT 2회(제주어·표준어, 공유 인코더).
각 인코더의 배치 순전파 처리량을 따로 잰 뒤 건당 시간을 더한다:
  items/s = 1 / Σ (passes_i / throughput_i)
처리량은 가중치 값과 무관하므로 원 저장소 재현 코드의 체크포인트가 없으면 공개 기반 가중치로 잰다.
LLM 쪽은 HTTP 서버 경유로 재므로 이 수치는 서버 오버헤드가 빠진 상한이다(README 한계 참고).

  python -m jeju_lora.bench_encoder --tag encoder_ensemble --gpu "RTX 4090"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from jeju_lora.data import read_jsonl
from jeju_lora.predict import DEFAULT_TEST, split_name

DEFAULT_ENCODERS = [
    # (이름, HF ID, 1건당 순전파 횟수, 입력 열)
    ("koelectra", "monologg/koelectra-base-v3-discriminator", 1, "jeju"),
    ("kr-bert-dual", "snunlp/KR-BERT-char16424", 2, "jeju+std"),
]


def ensemble_items_per_s(parts: list[tuple[int, float]]) -> float:
    """parts = [(passes, single-pass items/s), ...] → 앙상블 items/s."""
    if not parts or any(t <= 0 or p <= 0 for p, t in parts):
        raise ValueError("passes·throughput는 양수")
    return 1.0 / sum(p / t for p, t in parts)


def precision() -> str:
    """measure()가 쓰는 dtype 이름 (bf16이 없는 GPU는 fp16)."""
    import torch

    from jeju_lora.generation import auto_dtype, dtype_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return dtype_name(torch, auto_dtype(torch, device))


def measure(hf_id: str, texts: list[str], batch_size: int, max_length: int, warmup: int) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from jeju_lora.generation import auto_dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=7)
    model = model.to(device, dtype=auto_dtype(torch, device)).eval()

    def run(chunk_texts):
        for i in range(0, len(chunk_texts), batch_size):
            enc = tok(
                chunk_texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                model(**enc).logits.argmax(-1).cpu()

    run(texts[: warmup * batch_size])
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    run(texts)
    if device == "cuda":
        torch.cuda.synchronize()
    return len(texts) / (time.perf_counter() - t0)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_TEST)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=64, help="원 저장소 재현 코드의 기본값")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--tag", default="encoder_ensemble")
    p.add_argument("--gpu", default="")
    p.add_argument("--out-dir", default="reports/throughput")
    args = p.parse_args()

    items = read_jsonl(args.data)[: args.n]
    per_encoder, parts = {}, []
    for name, hf_id, passes, mode in DEFAULT_ENCODERS:
        texts = [it["jeju"] for it in items]
        if mode == "jeju+std":
            texts = texts + [it["std"] for it in items]
            # 2회 순전파를 한 목록으로 재고 1회당 처리량으로 환산한다
        tput = measure(hf_id, texts, args.batch_size, args.max_length, args.warmup)
        per_encoder[name] = {"hf_id": hf_id, "passes": passes, "single_pass_items_per_s": tput}
        parts.append((passes, tput))

    result = {
        "tag": args.tag,
        "system": "encoder",
        "items_per_s": ensemble_items_per_s(parts),
        "latency_p95_s": None,
        "batch_size": args.batch_size,
        "precision": precision(),
        "split": split_name(args.data),
        "n": len(items),
        "encoders": per_encoder,
        "gpu": args.gpu,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / f"{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"{args.tag}: {result['items_per_s']:.1f} items/s")


if __name__ == "__main__":
    main()

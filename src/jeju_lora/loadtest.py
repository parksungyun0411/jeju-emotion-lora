"""vLLM 서버 부하 측정: 동시 요청 수를 고정하고 건/s와 지연 분위수를 잰다.

요청 하나 = 평가 문항 하나(분류 1건). 처음 --warmup건은 집계에서 뺀다.

  python -m jeju_lora.loadtest --model qwen3-4b-awq --tag qwen3-4b_awq --concurrency 64
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import httpx

from jeju_lora.data import read_jsonl
from jeju_lora.generation import OpenAIChatBackend
from jeju_lora.predict import DEFAULT_TEST, split_name
from jeju_lora.prompts import build_messages


def percentile(values: list[float], q: float) -> float:
    """최근접 순위(nearest-rank) 분위수. q는 0~100."""
    if not values:
        raise ValueError("빈 목록")
    if not 0 < q <= 100:
        raise ValueError("q는 (0, 100]")
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def summarize(latencies: list[float], wall_s: float, errors: int) -> dict:
    n = len(latencies)
    return {
        "completed": n,
        "errors": errors,
        "wall_s": wall_s,
        "items_per_s": n / wall_s if wall_s > 0 else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
    }


async def run(
    backend: OpenAIChatBackend, messages: list[list[dict]], concurrency: int, warmup: int
):
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def one(client, m, record: bool):
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(backend.url, json=backend.payload(m))
                r.raise_for_status()
            except httpx.HTTPError:
                if record:
                    errors += 1
                return
            if record:
                latencies.append(time.perf_counter() - t0)

    async with httpx.AsyncClient(timeout=backend.timeout) as client:
        await asyncio.gather(*(one(client, m, False) for m in messages[:warmup]))
        t0 = time.perf_counter()
        await asyncio.gather(*(one(client, m, True) for m in messages[warmup:]))
        wall = time.perf_counter() - t0
    return summarize(latencies, wall, errors)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--data", default=DEFAULT_TEST)
    p.add_argument("--n", type=int, default=5000, help="측정 문항 수 (warmup 제외)")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--input-mode", default="jeju")
    p.add_argument("--gpu", default="", help="기록용 GPU 이름 (예: RTX 4090)")
    p.add_argument("--precision", default=None, help="기록용 서버 dtype (예: fp16, awq-w4a16)")
    p.add_argument("--out-dir", default="reports/throughput")
    args = p.parse_args()

    items = read_jsonl(args.data)[: args.warmup + args.n]
    messages = [build_messages(it, args.input_mode) for it in items]
    backend = OpenAIChatBackend(args.base_url, args.model)
    result = asyncio.run(run(backend, messages, args.concurrency, args.warmup))
    result.update(
        tag=args.tag,
        system="vllm",
        model=args.model,
        concurrency=args.concurrency,
        gpu=args.gpu,
        precision=args.precision,
        split=split_name(args.data),
        n_requested=args.n,
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / f"{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    p95_ms = result["latency_p95_s"] * 1000
    print(f"{args.tag}: {result['items_per_s']:.1f} items/s, p95 {p95_ms:.0f} ms")


if __name__ == "__main__":
    main()

"""결과·건당 비용 표 생성. GPU 시간당 요금($/h)은 입력으로 받는다.

  $/100만 건 = GPU $/h ÷ (건/s × 3600) × 1,000,000

측정 파일이 없는 칸은 '측정 전'으로 둔다. 숫자를 지어내지 않는다.
report.yaml의 행에 `not_run: 사유`가 있으면 그 행은 '측정 안 함'으로 두고 사유를 표 아래에 적는다.
`no_human: true`인 행은 독립 재라벨 열을 '—'로 둔다
(재라벨 문항을 뽑은 분할과 그 행의 분할이 다를 때).
분할·정밀도·백엔드 열은 측정 파일에 기록된 값을 그대로 옮긴다.

  python -m jeju_lora.cost --gpu-usd-per-hour 0.69 --gpu-name "RTX 4090"
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PENDING = "측정 전"
NOT_RUN = "측정 안 함"
SPLIT_NAMES = {"splits_sentence": "문장 단위", "splits": "원 행 단위"}
BACKEND_NAMES = {
    "openai": "vLLM",
    "hf": "transformers 생성",
    "hf-score": "transformers 라벨 점수",
    "encoder": "PyTorch 배치",
}
SECONDS_PER_HOUR = 3600


def usd_per_million(gpu_usd_per_hour: float, items_per_s: float) -> float:
    if gpu_usd_per_hour < 0:
        raise ValueError("요금은 0 이상")
    if items_per_s <= 0:
        raise ValueError("처리량은 양수")
    return gpu_usd_per_hour / (items_per_s * SECONDS_PER_HOUR) * 1_000_000


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value, spec: str) -> str:
    return PENDING if value is None else format(value, spec)


def _field(m: dict | None, t: dict | None, key: str):
    """지표 파일 값을 먼저, 없으면 처리량 파일 값을 쓴다."""
    for d in (m, t):
        if d and d.get(key) is not None:
            return d[key]
    return None


def _backend(m: dict | None, t: dict | None) -> str | None:
    name = (m or {}).get("backend") or (t or {}).get("system")
    if name is None:
        return None
    label = BACKEND_NAMES.get(name, name)
    if name == "openai" and m and not m.get("constrained"):
        label += " (제약 없음)"
    return label


def build_rows(report_cfg: dict, reports_dir: Path, gpu_usd_per_hour: float | None) -> list[dict]:
    rows = []
    for r in report_cfg["rows"]:
        if r.get("not_run"):
            rows.append({"name": r["name"], "not_run": r["not_run"]})
            continue
        m = _load(reports_dir / "metrics" / f"{r['metrics']}.json") if r.get("metrics") else None
        t = (
            _load(reports_dir / "throughput" / f"{r['throughput']}.json")
            if r.get("throughput")
            else None
        )
        human = m.get("human_label") if m else None
        ips = t.get("items_per_s") if t else None
        split = _field(m, t, "split")
        rows.append(
            {
                "name": r["name"],
                "split": SPLIT_NAMES.get(split, split),
                "precision": _field(m, t, "precision"),
                "backend": _backend(m, t),
                "f1_machine": m["machine_label"]["f1_macro"] if m else None,
                "f1_human": human["f1_macro"] if human else None,
                "invalid_rate": m["machine_label"]["invalid_rate"] if m else None,
                "items_per_s": ips,
                "p95_ms": t["latency_p95_s"] * 1000 if t and t.get("latency_p95_s") else None,
                "usd_per_million": (
                    usd_per_million(gpu_usd_per_hour, ips)
                    if ips and gpu_usd_per_hour is not None
                    else None
                ),
                "has_throughput_row": bool(r.get("throughput")),
                "no_human": bool(r.get("no_human")),
            }
        )
    return rows


def render_markdown(
    rows: list[dict], gpu_usd_per_hour: float | None, gpu_name: str, price_note: str = ""
) -> str:
    rate = f"${gpu_usd_per_hour:.4g}/h" if gpu_usd_per_hour is not None else PENDING
    lines = [f"GPU: {gpu_name or PENDING} · 요금 {rate}", ""]
    if price_note:
        lines += [f"요금 근거: {price_note}", ""]
    lines += [
        "| 구성 | 분할 | 정밀도 | 백엔드 | macro-F1 (기계 라벨) | macro-F1 (독립 재라벨) "
        "| 파싱 실패율 | 건/s | p95 (ms) | $/100만 건 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    reasons = []
    for r in rows:
        if "not_run" in r:
            reasons.append(f"- {r['name']}: {r['not_run']}")
            lines.append("| " + " | ".join([r["name"]] + [NOT_RUN] * 9) + " |")
            continue
        tp = r["has_throughput_row"]
        lines.append(
            "| "
            + " | ".join(
                [
                    r["name"],
                    r["split"] or PENDING,
                    r["precision"] or PENDING,
                    r["backend"] or PENDING,
                    _fmt(r["f1_machine"], ".4f"),
                    "—" if r["no_human"] else _fmt(r["f1_human"], ".4f"),
                    _fmt(r["invalid_rate"], ".4f"),
                    _fmt(r["items_per_s"], ".1f") if tp else "—",
                    _fmt(r["p95_ms"], ".0f")
                    if tp and (r["p95_ms"] or r["items_per_s"] is None)
                    else "—",  # 인코더 배치 측정은 p95 없음
                    _fmt(r["usd_per_million"], ".2f") if tp else "—",
                ]
            )
            + " |"
        )
    if reasons:
        lines += ["", f"{NOT_RUN} 사유:", "", *reasons]
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-usd-per-hour", type=float, default=None)
    p.add_argument("--gpu-name", default="")
    p.add_argument("--price-note", default="", help="요금 출처 (예: Colab 종량제 단가 × T4 소비율)")
    p.add_argument("--config", default="configs/report.yaml")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--out", default="reports/RESULTS.md")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = build_rows(cfg, Path(args.reports_dir), args.gpu_usd_per_hour)
    table = render_markdown(rows, args.gpu_usd_per_hour, args.gpu_name, args.price_note)
    header = (
        "# 결과\n\n`make report`가 reports/metrics·reports/throughput에서 생성한다. "
        "손으로 고치지 않는다.\n\n"
    )
    Path(args.out).write_text(header + table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

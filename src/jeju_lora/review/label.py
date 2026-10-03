"""터미널 라벨링 도구. 결정 하나마다 reviews/decisions.jsonl에 한 줄을 바로 추가한다.

기본은 블라인드(기계 라벨을 보여 주지 않음)다. 중단 후 다시 실행하면 이어서 한다.
같은 문항을 다시 판정하면 새 줄이 추가되고 보고서는 마지막 판정을 쓴다.

입력: 0~6 라벨 번호 [메모] · a [메모] 판단 불가 · s 건너뜀 · q 종료

  python -m jeju_lora.review.label --reviewer 박성윤
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jeju_lora.data import read_jsonl
from jeju_lora.labels import EMOTION_LABELS, NUM_LABELS

HELP = (
    "  ".join(f"{i}={name}" for i, name in EMOTION_LABELS.items()) + "  a=판단불가 s=건너뜀 q=종료"
)


def parse_input(text: str) -> tuple[str, int | None, str]:
    """→ (action, label, note). action ∈ label|ambiguous|skip|quit|invalid"""
    s = text.strip()
    if not s:
        return "invalid", None, ""
    head, _, note = s.partition(" ")
    note = note.strip()
    if head in {"q", "quit"}:
        return "quit", None, ""
    if head in {"s", "skip"}:
        return "skip", None, ""
    if head == "a":
        return "ambiguous", None, note
    if head.isdigit() and 0 <= int(head) < NUM_LABELS:
        return "label", int(head), note
    return "invalid", None, ""


def pending(items: list[dict], decisions: list[dict], reviewer: str) -> list[dict]:
    done = {d["row_id"] for d in decisions if d["reviewer"] == reviewer}
    return [it for it in items if it["row_id"] not in done]


def make_record(
    item: dict, reviewer: str, action: str, label: int | None, note: str, shown: bool, now=None
) -> dict:
    return {
        "row_id": item["row_id"],
        "reviewer": reviewer,
        "decided_at": (now or datetime.now().astimezone()).isoformat(timespec="seconds"),
        "decision": action,
        "human_label": label,
        "machine_label_shown": shown,
        "note": note,
    }


def run(items, decisions_path: Path, reviewer: str, show_machine: bool, ask=input, say=print):
    decisions = read_jsonl(decisions_path) if decisions_path.exists() else []
    todo = pending(items, decisions, reviewer)
    say(f"{reviewer}: 남은 {len(todo)}건 / 전체 {len(items)}건")
    say(HELP)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with decisions_path.open("a", encoding="utf-8") as f:
        for k, it in enumerate(todo, 1):
            say(f"\n[{k}/{len(todo)}] row_id={it['row_id']}")
            say(f"  제주어: {it['jeju']}")
            say(f"  표준어: {it['std']}")
            if show_machine:
                say(f"  기계 라벨: {EMOTION_LABELS[it['machine_label']]}")
            while True:
                action, label, note = parse_input(ask("> "))
                if action != "invalid":
                    break
                say(HELP)
            if action == "quit":
                break
            if action == "skip":
                continue
            rec = make_record(it, reviewer, action, label, note, show_machine)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
    say(f"\n이번에 {written}건 기록 → {decisions_path}")
    return written


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", default="data/review/sample.jsonl")
    p.add_argument("--out", default="reviews/decisions.jsonl")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--show-machine-label", action="store_true")
    args = p.parse_args()
    run(read_jsonl(args.items), Path(args.out), args.reviewer, args.show_machine_label)


if __name__ == "__main__":
    main()

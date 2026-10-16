"""검수 결과 집계: 재라벨 대 GPT-4o 일치도, 검수자 간 일치도, 최종 gold 파일.

- 검수자·문항별로 마지막 판정만 쓴다.
- '판단 불가'는 gold에서 뺀다.
- 검수자가 여럿이면 다수결, 동률이면 gold에서 뺀다.

  python -m jeju_lora.review.report
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.labels import LABEL_NAMES
from jeju_lora.metrics import cohen_kappa


def latest(decisions: list[dict]) -> dict[tuple[str, int], dict]:
    out = {}
    for d in sorted(decisions, key=lambda d: d["decided_at"]):
        out[(d["reviewer"], int(d["row_id"]))] = d
    return out


def build_gold(final: dict[tuple[str, int], dict]) -> list[dict]:
    votes: dict[int, list[int]] = {}
    for (_, row_id), d in final.items():
        if d["decision"] == "label":
            votes.setdefault(row_id, []).append(int(d["human_label"]))
    gold = []
    for row_id, vs in sorted(votes.items()):
        top = Counter(vs).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue
        gold.append({"row_id": row_id, "human_label": top[0][0], "n_reviewers": len(vs)})
    return gold


def agreement(final, machine: dict[int, int]) -> dict:
    reviewers = sorted({r for r, _ in final})
    per_reviewer = {}
    for rv in reviewers:
        ds = [d for (r, _), d in final.items() if r == rv]
        labelled = [d for d in ds if d["decision"] == "label"]
        h = [int(d["human_label"]) for d in labelled]
        m = [machine[int(d["row_id"])] for d in labelled]
        per_class = {}
        for c, name in enumerate(LABEL_NAMES):
            idx = [i for i, x in enumerate(m) if x == c]
            per_class[name] = sum(h[i] == c for i in idx) / len(idx) if idx else None
        per_reviewer[rv] = {
            "decided": len(ds),
            "ambiguous": sum(d["decision"] == "ambiguous" for d in ds),
            "labelled": len(labelled),
            "agree_with_gpt4o": (sum(a == b for a, b in zip(h, m, strict=True)) / len(h))
            if h
            else None,
            "kappa_vs_gpt4o": cohen_kappa(h, m) if h else None,
            "agree_by_machine_class": per_class,
        }
    pairs = {}
    for a, b in combinations(reviewers, 2):
        la = {rid: d for (r, rid), d in final.items() if r == a and d["decision"] == "label"}
        lb = {rid: d for (r, rid), d in final.items() if r == b and d["decision"] == "label"}
        shared = sorted(set(la) & set(lb))
        pairs[f"{a}|{b}"] = {
            "shared": len(shared),
            "kappa": (
                cohen_kappa(
                    [la[i]["human_label"] for i in shared], [lb[i]["human_label"] for i in shared]
                )
                if shared
                else None
            ),
        }
    return {"reviewers": per_reviewer, "inter_reviewer": pairs}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decisions", default="reviews/decisions.jsonl")
    p.add_argument("--items", default="data/review/sample.jsonl")
    p.add_argument("--gold-out", default="reviews/gold.jsonl")
    p.add_argument("--report-out", default="reviews/agreement.json")
    args = p.parse_args()

    if not Path(args.decisions).exists():
        raise SystemExit(f"{args.decisions} 없음 — 아직 검수 전이다")
    items = read_jsonl(args.items)
    machine = {int(it["row_id"]): int(it["machine_label"]) for it in items}
    final = latest(read_jsonl(args.decisions))
    gold = build_gold(final)
    rep = {"sample_size": len(items), "gold_size": len(gold), **agreement(final, machine)}
    write_jsonl(gold, args.gold_out)
    Path(args.report_out).write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

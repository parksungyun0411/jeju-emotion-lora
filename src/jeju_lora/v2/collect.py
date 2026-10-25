"""라벨러 출력 검증·모으기·두 번 라벨링 일치도.

    python -m jeju_lora.v2.collect check --batches data/v2/batches/evalA \
        --labels data/v2/labels/evalA
    python -m jeju_lora.v2.collect agree --a data/v2/labels/evalA --b data/v2/labels/evalB \
        [--ids data/v2/items/test.jsonl]

check는 배치마다 id 집합·순서·값 범위를 대조하고, 문제가 있는 배치 번호를 낸다(다시 돌릴 대상).
agree는 두 라벨 묶음의 일치율·Cohen's κ와, 한쪽을 정답으로 다른 쪽을 예측으로 본 macro-F1(양쪽
평균)을 낸다. 이 macro-F1은 같은 지침의 라벨러가 다시 붙였을 때 맞출 수 있는 정도라서, 모델
점수의 상한 참고값으로 쓴다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score

from jeju_lora.labels import LABEL_NAMES, NUM_LABELS

CONFS = {"high", "mid", "low"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


PART_SIZE = 50
HEAD_CHARS = 8


def part_paths(batch: Path, label_dir: Path) -> list[Path]:
    """배치 NNN.jsonl의 라벨은 NNN.p1.jsonl, NNN.p2.jsonl … (50줄씩)."""
    n_items = sum(1 for line in batch.open(encoding="utf-8") if line.strip())
    n_parts = -(-n_items // PART_SIZE)
    return [label_dir / f"{batch.stem}.p{k}.jsonl" for k in range(1, n_parts + 1)]


def _norm(text: str) -> str:
    return "".join(str(text).split())


def check_batch(batch: Path, label_dir: Path) -> list[str]:
    """문제 목록. 비어 있으면 통과.

    id 집합·순서·값 범위에 더해, 줄마다 `head`가 그 id 문장의 앞 8글자와 같은지 본다(공백 무시).
    라벨러가 긴 출력에서 한 줄씩 밀려 쓰면 id는 맞아도 라벨이 옆 문장 것이 되는데, 이것을 잡는다.
    """
    out: list[dict] = []
    for p in part_paths(batch, label_dir):
        if not p.exists():
            return [f"missing {p.name}"]
        try:
            out.extend(read_jsonl(p))
        except json.JSONDecodeError as e:
            return [f"bad json in {p.name}: {e}"]
    items = read_jsonl(batch)
    want = [r["id"] for r in items]
    got = [r.get("id") for r in out]
    problems = []
    heads = {r["id"]: _norm(r["jeju"][:HEAD_CHARS]) for r in items}
    off = [r for r in out if r.get("id") in heads and _norm(r.get("head", "")) != heads[r["id"]]]
    if off:
        problems.append(f"head mismatch: {len(off)}")
    if set(got) != set(want):
        missing, extra = len(set(want) - set(got)), len(set(got) - set(want))
        problems.append(f"ids: missing {missing}, extra {extra}")
    elif got != want:
        problems.append("order differs")
    if len(got) != len(set(got)):
        problems.append(f"duplicate ids: {len(got) - len(set(got))}")
    bad = [r for r in out if r.get("label") not in range(NUM_LABELS) or r.get("conf") not in CONFS]
    if bad:
        problems.append(f"invalid label/conf: {len(bad)}")
    return problems


def load_labels(label_dir: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for p in sorted(label_dir.glob("*.jsonl")):
        for r in read_jsonl(p):
            labels[r["id"]] = r
    return labels


def agreement(a: dict[str, dict], b: dict[str, dict], ids: list[str] | None = None) -> dict:
    common = [i for i in (ids if ids is not None else a) if i in a and i in b]
    ya = [a[i]["label"] for i in common]
    yb = [b[i]["label"] for i in common]
    labels = list(range(NUM_LABELS))
    f1_ab = f1_score(ya, yb, labels=labels, average="macro", zero_division=0)
    f1_ba = f1_score(yb, ya, labels=labels, average="macro", zero_division=0)
    per_class = f1_score(ya, yb, labels=labels, average=None, zero_division=0)
    by_conf = {}
    for c in ("high", "mid", "low"):
        sel = [k for k, i in enumerate(common) if a[i]["conf"] == c and b[i]["conf"] == c]
        if sel:
            by_conf[c] = {"n": len(sel), "agree": sum(ya[k] == yb[k] for k in sel) / len(sel)}
    return {
        "n": len(common),
        "agree": sum(x == y for x, y in zip(ya, yb, strict=True)) / max(len(common), 1),
        "kappa": cohen_kappa_score(ya, yb, labels=labels),
        "macro_f1_mean": (f1_ab + f1_ba) / 2,
        "per_class_f1": {LABEL_NAMES[k]: round(float(v), 4) for k, v in enumerate(per_class)},
        "dist_a": {LABEL_NAMES[k]: v for k, v in sorted(Counter(ya).items())},
        "dist_b": {LABEL_NAMES[k]: v for k, v in sorted(Counter(yb).items())},
        "agree_when_both_conf": by_conf,
    }


def finalize(data: Path, labels_root: Path) -> dict:
    """확정 라벨 → data/final/{split}.jsonl, LoRA 코드용 data/../splits_v2/{split}.jsonl.

    train: 한 번 라벨링한 값. val·test: evalA·evalB가 같으면 그 값, 다르면 evalC(불일치만 다시
    붙인 세 번째 라벨)와 다수결. 셋이 모두 다르면 그 문장은 뺀다(수를 manifest에 남긴다).
    """

    def passes(*names: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for n in names:
            if (labels_root / n).exists():
                out.update(load_labels(labels_root / n))
        return out

    # 자연 분포(eval*)와 보강셋(cue*)은 배치가 다르고 id는 겹치지 않는다.
    a, b, c = passes("evalA", "cueA"), passes("evalB", "cueB"), passes("evalC", "cueC")
    train = load_labels(labels_root / "train")
    stats: dict = {}
    (data / "final").mkdir(parents=True, exist_ok=True)
    export_dir = data.parent / "splits_v2"
    export_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "val_cue", "test_cue"):
        path = data / "items" / f"{split}.jsonl"
        if not path.exists():
            continue
        items = read_jsonl(path)
        out, why = [], Counter()
        for it in items:
            i = it["id"]
            if split == "train":
                if i not in train:
                    why["unlabeled"] += 1
                    continue
                lab = train[i]["label"]
            else:
                if i not in a or i not in b:
                    why["unlabeled"] += 1
                    continue
                if a[i]["label"] == b[i]["label"]:
                    lab, why["agreed"] = a[i]["label"], why["agreed"] + 1
                elif i in c and c[i]["label"] in (a[i]["label"], b[i]["label"]):
                    lab, why["majority"] = c[i]["label"], why["majority"] + 1
                else:
                    why["no_majority" if i in c else "pending_adjudication"] += 1
                    continue
            out.append({"id": i, "label": int(lab)})
        with (data / "final" / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
        by_id = {it["id"]: it for it in items}
        with (export_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row_id, r in enumerate(out):
                it = by_id[r["id"]]
                rec = {
                    "row_id": row_id,
                    "id": r["id"],
                    "jeju": it["jeju"],
                    "std": it["std"],
                    "label": r["label"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dist = Counter(LABEL_NAMES[r["label"]] for r in out)
        stats[split] = {"kept": len(out), **why, "dist": dict(dist)}
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--batches", required=True, type=Path)
    c.add_argument("--labels", required=True, type=Path)
    g = sub.add_parser("agree")
    g.add_argument("--a", required=True, type=Path)
    g.add_argument("--b", required=True, type=Path)
    g.add_argument("--ids", type=Path, help="이 items 파일의 id만 (예: items/test.jsonl)")
    fz = sub.add_parser("finalize")
    fz.add_argument("--data", type=Path, default=Path("data/v2"))
    fz.add_argument("--labels", type=Path, default=Path("data/v2/labels"))
    args = ap.parse_args(argv)

    if args.cmd == "finalize":
        stats = finalize(args.data, args.labels)
        (args.data / "final" / "manifest.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.cmd == "check":
        failed = {}
        batches = sorted(args.batches.glob("*.jsonl"))
        for b in batches:
            problems = check_batch(b, args.labels)
            if problems:
                failed[b.stem] = problems
        print(json.dumps({"batches": len(batches), "failed": failed}, ensure_ascii=False, indent=2))
    else:
        a = load_labels(args.a) if args.a.is_dir() else {r["id"]: r for r in read_jsonl(args.a)}
        b = load_labels(args.b) if args.b.is_dir() else {r["id"]: r for r in read_jsonl(args.b)}
        ids = [r["id"] for r in read_jsonl(args.ids)] if args.ids else None
        print(json.dumps(agreement(a, b, ids), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

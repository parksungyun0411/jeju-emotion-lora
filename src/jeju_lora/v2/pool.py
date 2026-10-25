"""AI Hub 제주 발화 → 문장 풀 → 대화 파일 단위 분할 → 표본 → 라벨링 배치.

    python -m jeju_lora.v2.pool --zip "<[라벨]제주도_학습용데이터_1.zip>" --out data/v2

분할은 대화 파일 단위다. 파일 이름의 해시로 train 80% · val 10% · test 10%를 정해, 같은 대화와
화자가 두 분할에 함께 들어가지 않는다. 문장은 분할마다 중복을 없애고, val·test에서는 train에
있는 문장을 뺀다.

표본: val·test는 각 분할 풀에서 무작위(자연 분포). train은 무작위 70% + 감정 단서어가 있는
문장 30%(단서어 묶음마다 같은 수)로, 드문 감정의 학습 예시를 늘린다. 단서어는 표본을 뽑을 때만
쓰고 라벨에는 쓰지 않는다. val·test에는 쓰지 않는다.

출력(data/ 아래라 git에 올라가지 않는다. 원천 문장은 AI Hub 라이선스를 따른다):
  items/{train,val,test}.jsonl   id, file, split, ctx_before, jeju, std, ctx_after
  batches/{pass}/{NNN}.jsonl     라벨러 입력. pass는 train, evalA, evalB
  manifest.json                  수·시드·규칙
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

SEED = 42
SPLIT_RATIO = {"train": 0.8, "val": 0.1}  # 나머지가 test
MIN_LEN, MAX_LEN = 10, 80
N_SAMPLE = {"train": 50_000, "val": 5_000, "test": 5_000}
# 보강 평가셋: val·test 대화 파일에서 단서어 묶음마다 같은 수. 자연 분포 test에는 공포·혐오가
# 1% 안팎이라 감정별 F1이 몇십 문장에 좌우되므로, 드문 감정 쪽 추정을 보태는 보조 지표로 쓴다.
N_CUE_EVAL = {"val_cue": 1_500, "test_cue": 2_500}
TRAIN_CUE_FRACTION = 0.3
BATCH_SIZE = 400
CTX_BEFORE, CTX_AFTER = 2, 1

# (방언)/(표준어) 쌍 표기 → 방언 쪽. '#'은 전사자의 잡음 표시라 지운다.
_PAIR_RE = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")
_TAG_RE = re.compile(r"\{[^}]*\}")
_ANON_RE = re.compile(r"&[^&\s]+&")
# 알아듣지 못한 구간·음절이 있는 발화는 문장으로 쓰지 않는다.
_UNCLEAR_RE = re.compile(r"\(\(|\)\)|\*|@")
_SPACE_RE = re.compile(r"\s+")

# 표본 보강용 단서어(표준어 풀이에서 찾는다). 라벨과는 무관하다.
CUES: dict[str, tuple[str, ...]] = {
    "행복": ("좋", "행복", "재밌", "재미있", "신나", "웃기", "맛있", "기쁘", "고마"),
    "슬픔": ("슬프", "서럽", "아쉽", "그립", "우울", "속상", "외롭", "눈물", "울었"),
    "분노": ("짜증", "화나", "화가", "열받", "빡", "미친", "어이없", "억울"),
    "놀람": ("깜짝", "놀라", "놀랐", "세상에", "대박", "헐", "어머"),
    "공포": ("무서", "무섭", "겁", "걱정", "불안", "위험", "떨려"),
    "혐오": ("더럽", "역겹", "징그", "싫", "토 나", "냄새", "구역"),
}


def clean_dialect(text: str) -> str | None:
    """dialect_form → 방언 문장. 쓸 수 없는 발화면 None."""
    if _UNCLEAR_RE.search(_PAIR_RE.sub("", text)):
        return None
    t = _PAIR_RE.sub(lambda m: m.group(1), text)
    t = _TAG_RE.sub(" ", t)
    t = _ANON_RE.sub("OO", t)
    t = t.replace("#", "")
    if "(" in t or ")" in t or "/" in t:
        return None
    t = _SPACE_RE.sub(" ", t).strip()
    return t or None


def clean_standard(text: str) -> str:
    t = _TAG_RE.sub(" ", text)
    t = _ANON_RE.sub("OO", t)
    return _SPACE_RE.sub(" ", t.replace("#", "")).strip()


def split_of(file_name: str, seed: int = SEED) -> str:
    h = int(hashlib.sha256(f"{seed}:{file_name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < SPLIT_RATIO["train"]:
        return "train"
    if h < SPLIT_RATIO["train"] + SPLIT_RATIO["val"]:
        return "val"
    return "test"


def _ctx_line(u: dict) -> str:
    d = clean_dialect(u.get("dialect_form") or u.get("form") or "") or "(알아듣기 어려운 발화)"
    return f"{u.get('speaker_id', '?')}: {d}"


def iter_candidates(zip_path: Path):
    """대화 파일마다 쓸 수 있는 문장 후보를 문맥과 함께 낸다."""
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".json")):
            utts = json.loads(z.read(name))["utterance"]
            file_id = Path(name).stem
            for i, u in enumerate(utts):
                jeju = clean_dialect(u.get("dialect_form") or "")
                if jeju is None or not (MIN_LEN <= len(jeju) <= MAX_LEN):
                    continue
                std = clean_standard(u.get("standard_form") or "")
                if _SPACE_RE.sub("", jeju) == _SPACE_RE.sub("", std):
                    continue  # 표준어와 같은 발화는 제주어 예시가 아니다
                yield {
                    "id": f"{file_id}.{i}",
                    "file": file_id,
                    "jeju": jeju,
                    "std": std,
                    "ctx_before": [_ctx_line(x) for x in utts[max(0, i - CTX_BEFORE) : i]],
                    "ctx_after": [_ctx_line(x) for x in utts[i + 1 : i + 1 + CTX_AFTER]],
                    "speaker": u.get("speaker_id", "?"),
                }


def cue_group(std: str) -> str | None:
    for group, words in CUES.items():
        if any(w in std for w in words):
            return group
    return None


def build_pools(zip_path: Path, seed: int = SEED) -> dict[str, list[dict]]:
    pools: dict[str, dict[str, dict]] = {"train": {}, "val": {}, "test": {}}
    for c in iter_candidates(zip_path):
        s = split_of(c["file"], seed)
        c["split"] = s
        pools[s].setdefault(c["jeju"], c)  # 분할 안에서 같은 문장은 처음 것만
    train_sentences = set(pools["train"])
    out = {"train": list(pools["train"].values())}
    for s in ("val", "test"):
        out[s] = [c for j, c in pools[s].items() if j not in train_sentences]
    return out


def sample(pools: dict[str, list[dict]], seed: int = SEED) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    chosen: dict[str, list[dict]] = {}
    for s in ("val", "test"):
        chosen[s] = rng.sample(pools[s], N_SAMPLE[s])
    pool = pools["train"][:]
    rng.shuffle(pool)
    n_cue = int(N_SAMPLE["train"] * TRAIN_CUE_FRACTION)
    per_group = n_cue // len(CUES)
    picked: list[dict] = []
    taken: set[str] = set()
    by_group: dict[str, list[dict]] = {g: [] for g in CUES}
    for c in pool:
        g = cue_group(c["std"])
        if g is not None:
            by_group[g].append(c)
    for g, items in by_group.items():
        for c in items[:per_group]:
            c["sampled_by"] = f"cue:{g}"
            picked.append(c)
            taken.add(c["id"])
    for c in pool:
        if len(picked) >= N_SAMPLE["train"]:
            break
        if c["id"] not in taken:
            c["sampled_by"] = "random"
            picked.append(c)
            taken.add(c["id"])
    rng.shuffle(picked)
    chosen["train"] = picked
    for s in ("val", "test"):
        for c in chosen[s]:
            c["sampled_by"] = "random"
    return chosen


def sample_cue_eval(
    pools: dict[str, list[dict]], chosen: dict[str, list[dict]], seed: int = SEED
) -> dict[str, list[dict]]:
    """자연 분포 표본과 겹치지 않게, 단서어 묶음마다 같은 수를 뽑는다(별도 난수)."""
    out = {}
    for name, n in N_CUE_EVAL.items():
        base = name.removesuffix("_cue")
        taken = {c["id"] for c in chosen[base]}
        rng = random.Random(seed + 100 + len(out))
        by_group: dict[str, list[dict]] = {g: [] for g in CUES}
        for c in pools[base]:
            g = cue_group(c["std"])
            if g is not None and c["id"] not in taken:
                by_group[g].append(c)
        per = n // len(CUES)
        picked = []
        for g, items in by_group.items():
            for c in rng.sample(items, min(per, len(items))):
                picked.append({**c, "sampled_by": f"cue:{g}"})
        rng.shuffle(picked)
        out[name] = picked
    return out


def labeler_view(c: dict) -> dict:
    """라벨러에게 보이는 필드만. 분할·표본 방식은 보이지 않는다."""
    return {k: c[k] for k in ("id", "ctx_before", "jeju", "std", "ctx_after", "speaker")}


def write_batches(items: list[dict], out_dir: Path, size: int = BATCH_SIZE) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for n, start in enumerate(range(0, len(items), size), start=1):
        with (out_dir / f"{n:03d}.jsonl").open("w", encoding="utf-8") as f:
            for c in items[start : start + size]:
                f.write(json.dumps(labeler_view(c), ensure_ascii=False) + "\n")
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/v2"), type=Path)
    args = ap.parse_args(argv)

    pools = build_pools(args.zip)
    chosen = sample(pools)
    chosen.update(sample_cue_eval(pools, chosen))
    (args.out / "items").mkdir(parents=True, exist_ok=True)
    for s, items in chosen.items():
        with (args.out / "items" / f"{s}.jsonl").open("w", encoding="utf-8") as f:
            for c in items:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # val·test는 두 번 라벨링한다. 두 번째는 다른 순서·다른 묶음으로 섞어,
    # 앞 묶음의 흐름을 따라가지 않게 한다.
    evals = chosen["val"] + chosen["test"]
    pass_a = evals[:]
    random.Random(SEED + 1).shuffle(pass_a)
    pass_b = evals[:]
    random.Random(SEED + 2).shuffle(pass_b)
    n_batches = {
        "train": write_batches(chosen["train"], args.out / "batches" / "train"),
        "evalA": write_batches(pass_a, args.out / "batches" / "evalA"),
        "evalB": write_batches(pass_b, args.out / "batches" / "evalB"),
    }
    cues = chosen["val_cue"] + chosen["test_cue"]
    for k, tag in ((3, "cueA"), (4, "cueB")):
        order = cues[:]
        random.Random(SEED + k).shuffle(order)
        n_batches[tag] = write_batches(order, args.out / "batches" / tag)
    manifest = {
        "seed": SEED,
        "split_unit": "conversation file (sha256 of seed:file)",
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "length_chars": [MIN_LEN, MAX_LEN],
        "pool_sentences": {s: len(p) for s, p in pools.items()},
        "pool_files": {s: len({c["file"] for c in p}) for s, p in pools.items()},
        "sampled": {s: len(v) for s, v in chosen.items()},
        "train_sampled_by": dict(Counter(c["sampled_by"] for c in chosen["train"])),
        "cue_eval_sampled_by": {
            k: dict(Counter(c["sampled_by"] for c in chosen[k])) for k in N_CUE_EVAL
        },
        "batches": n_batches,
        "batch_size": BATCH_SIZE,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (args.out / "manifest.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

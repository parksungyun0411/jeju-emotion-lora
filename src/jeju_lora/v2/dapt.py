"""발표본 4·5단계: 제주어 토큰 추가 + DAPT(제주어 MLM 재학습).

    python -m jeju_lora.v2.dapt corpus --zip "<라벨 zip>" --out data/v2/corpus_train.txt
    python -m jeju_lora.v2.dapt train --base snunlp/KR-BERT-char16424 \
        --corpus data/v2/corpus_train.txt --out runs/v2/dapt_krbert

말뭉치는 **train 대화 파일의 발화만** 쓴다(라벨 없음). val·test 대화 파일은 분할 규칙
(pool.split_of)으로 빼므로, DAPT가 평가 문장을 미리 보는 일은 없다.

토큰 추가: 말뭉치 어절 가운데 빈도가 min_freq 이상이고 기존 토크나이저가 3조각 이상으로 쪼개는
것을 빈도순으로 top_n개 더한다. 새 임베딩은 그 어절을 이루던 조각 임베딩의 평균으로 둔다.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from jeju_lora.v2.pool import SEED, clean_dialect, split_of


def export_corpus(zip_path: Path, out: Path) -> int:
    import json
    import zipfile

    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as z, out.open("w", encoding="utf-8") as f:
        for name in sorted(n for n in z.namelist() if n.endswith(".json")):
            if split_of(Path(name).stem, SEED) != "train":
                continue
            for u in json.loads(z.read(name))["utterance"]:
                t = clean_dialect(u.get("dialect_form") or "")
                if t and len(t) >= 4 and t not in seen:
                    seen.add(t)
                    f.write(t + "\n")
    return len(seen)


def pick_new_tokens(tok, lines: list[str], top_n: int, min_freq: int) -> list[str]:
    freq = Counter(w for line in lines for w in line.split())
    vocab = tok.get_vocab()
    picked = []
    for w, c in freq.most_common():
        if c < min_freq or len(picked) >= top_n:
            break
        if w not in vocab and len(tok.tokenize(w)) >= 3:
            picked.append(w)
    return picked


def train_mlm(args) -> None:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    lines = [x for x in args.corpus.read_text(encoding="utf-8").splitlines() if x.strip()]
    random.Random(SEED).shuffle(lines)
    if args.max_lines:
        lines = lines[: args.max_lines]
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForMaskedLM.from_pretrained(args.base)

    new_tokens = pick_new_tokens(tok, lines, args.top_n, args.min_freq) if args.top_n else []
    if new_tokens:
        pieces = {w: tok(w, add_special_tokens=False)["input_ids"] for w in new_tokens}
        tok.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tok))
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            for w in new_tokens:
                emb[tok.convert_tokens_to_ids(w)] = emb[pieces[w]].mean(0)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "added_jeju_tokens.txt").write_text("\n".join(new_tokens) + "\n", encoding="utf-8")

    ds = Dataset.from_dict({"text": lines}).map(
        lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    targs = TrainingArguments(
        output_dir=str(args.out / "trainer"),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=0.06,  # transformers 5: 1 미만 실수는 비율
        weight_decay=0.01,
        logging_steps=200,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved {args.out} (+{len(new_tokens)} tokens, {len(lines)} lines)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--zip", required=True, type=Path)
    c.add_argument("--out", type=Path, default=Path("data/v2/corpus_train.txt"))
    t = sub.add_parser("train")
    t.add_argument("--base", required=True)
    t.add_argument("--corpus", type=Path, default=Path("data/v2/corpus_train.txt"))
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--top-n", type=int, default=1000)
    t.add_argument("--min-freq", type=int, default=200)
    t.add_argument("--max-lines", type=int, default=0)
    t.add_argument("--max-len", type=int, default=64)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args(argv)
    if args.cmd == "corpus":
        print(f"{export_corpus(args.zip, args.out)} lines → {args.out}")
    else:
        train_mlm(args)


if __name__ == "__main__":
    main()

"""라벨 v2 인코더 학습: 제주어 단일 입력 또는 제주어+표준어 Dual-Gated.

    python -m jeju_lora.v2.train_encoder --encoder snunlp/KR-BERT-char16424 --input jeju \
        --data data/v2 --out runs/v2/krbert_jeju
    python -m jeju_lora.v2.train_encoder --encoder runs/v2/dapt_krbert --input dual ...

원 프로젝트 발표본의 학습 설정(LR 3e-5, batch 64, max length 80, dropout 0.2, AdamW wd 0.01,
warmup 20%, cosine, label smoothing 0.05, grad clip 1.0, hard mining)을 따르되 두 가지를 바꿨다.

- 체크포인트와 하이퍼파라미터는 val macro-F1로만 고른다. test는 마지막에 한 번 예측해 저장만 한다.
- v2 라벨은 자연 분포(중립이 대부분)라, 손실에 클래스 가중치(train 빈도의 역수 제곱근)를 준다.

`--input dual`은 전사자가 단 표준어 풀이를 입력으로 쓴다. 실제 서비스에는 표준어 풀이가 없으므로
결과를 따로 적는다.

출력: out/metrics.json (val·test 지표, 에폭별 val), out/{val,test}_probs.npz (id·probs·labels)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from jeju_lora.labels import LABEL_NAMES, NUM_LABELS

HP = {
    "lr": 3e-5,
    "batch_size": 64,
    "epochs": 7,
    "max_len": 80,
    "dropout": 0.2,
    "weight_decay": 0.01,
    "warmup": 0.2,
    "label_smoothing": 0.05,
    "grad_clip": 1.0,
    "hard_top": 0.2,  # hard mining: 손실 상위 20%
    "hard_alpha": 2.0,  # 그 문장들의 가중치
    "hard_from_epoch": 3,  # 이 에폭부터 적용(1부터 셈)
}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(data: Path, split: str, labels_file: Path | None = None) -> list[dict]:
    items = {}
    with (data / "items" / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items[r["id"]] = r
    path = labels_file or data / "final" / f"{split}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            lab = json.loads(line)
            if lab["id"] in items:
                rows.append({**items[lab["id"]], "label": int(lab["label"])})
    return rows


class Rows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {"idx": i, "jeju": r["jeju"], "std": r["std"], "label": r["label"]}


def make_collate(tok, max_len: int, dual: bool):
    def collate(batch: list[dict]) -> dict:
        out = {
            "idx": torch.tensor([b["idx"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch]),
        }
        enc = tok(
            [b["jeju"] for b in batch],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        out["jeju"] = dict(enc)
        if dual:
            enc2 = tok(
                [b["std"] for b in batch],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            out["std"] = dict(enc2)
        return out

    return collate


class Classifier(nn.Module):
    """dual=False: [CLS] → Linear. dual=True: 공유 인코더로 제주어·표준어 [CLS]를 얻어
    g = σ(W[h_j; h_s]), h = g·h_j + (1−g)·h_s (원 프로젝트 Dual-Gated와 같은 식)."""

    def __init__(self, encoder_name: str, dual: bool, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dual = dual
        self.gate = nn.Linear(2 * h, h) if dual else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, NUM_LABELS)

    def cls(self, enc: dict) -> torch.Tensor:
        return self.encoder(**enc).last_hidden_state[:, 0]

    def forward(self, batch: dict) -> torch.Tensor:
        hj = self.cls(batch["jeju"])
        if self.dual:
            hs = self.cls(batch["std"])
            g = torch.sigmoid(self.gate(torch.cat([hj, hs], dim=-1)))
            hj = g * hj + (1 - g) * hs
        return self.head(self.drop(hj))


def to_device(batch: dict, dev: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = {kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev)
    return out


def class_weights(rows: list[dict]) -> torch.Tensor:
    counts = np.bincount([r["label"] for r in rows], minlength=NUM_LABELS).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float)


@torch.no_grad()
def predict(model, loader, dev) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        b = to_device(batch, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(b)
        probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = probs.argmax(-1)
    per = f1_score(labels, pred, labels=list(range(NUM_LABELS)), average=None, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(labels, pred, labels=list(range(NUM_LABELS)), average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred)),
        "per_class_f1": {LABEL_NAMES[i]: round(float(v), 4) for i, v in enumerate(per)},
    }


def per_sample_loss(model, loader, dev, n: int, weight: torch.Tensor) -> np.ndarray:
    model.eval()
    losses = np.zeros(n)
    with torch.no_grad():
        for batch in loader:
            b = to_device(batch, dev)
            logits = model(b).float()
            loss = F.cross_entropy(logits, b["labels"], weight=weight, reduction="none")
            losses[batch["idx"].numpy()] = loss.cpu().numpy()
    return losses


def train(args) -> dict:
    hp = {**HP, **{k: v for k, v in vars(args).items() if k in HP and v is not None}}
    set_seed(args.seed)
    dev = device()
    dual = args.input == "dual"
    tr = load_split(args.data, "train")
    va = load_split(args.data, "val")
    te = load_split(args.data, "test")
    if args.limit:
        tr, va, te = tr[: args.limit], va[: args.limit // 4], te[: args.limit // 4]
    tok = AutoTokenizer.from_pretrained(args.encoder)
    collate = make_collate(tok, hp["max_len"], dual)
    train_ds = Rows(tr)
    eval_loader = lambda rows: DataLoader(Rows(rows), batch_size=128, collate_fn=collate)  # noqa: E731
    model = Classifier(args.encoder, dual, hp["dropout"]).to(dev)
    weight = class_weights(tr).to(dev)
    sample_w = np.ones(len(tr))

    steps_per_epoch = math.ceil(len(tr) / hp["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    total = steps_per_epoch * hp["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(total * hp["warmup"]), total)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    va_loader = eval_loader(va)
    for epoch in range(1, hp["epochs"] + 1):
        if hp["hard_alpha"] > 1 and epoch >= hp["hard_from_epoch"]:
            seq_loader = DataLoader(train_ds, batch_size=128, collate_fn=collate)
            losses = per_sample_loss(model, seq_loader, dev, len(tr), weight)
            cut = np.quantile(losses, 1 - hp["hard_top"])
            sample_w = np.where(losses >= cut, hp["hard_alpha"], 1.0)
        model.train()
        loader = DataLoader(
            train_ds,
            batch_size=hp["batch_size"],
            shuffle=True,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(epoch),
        )
        for batch in loader:
            b = to_device(batch, dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(b)
            loss = F.cross_entropy(
                logits.float(),
                b["labels"],
                weight=weight,
                label_smoothing=hp["label_smoothing"],
                reduction="none",
            )
            sw = torch.tensor(sample_w[batch["idx"].numpy()], dtype=torch.float, device=dev)
            loss = (loss * sw).sum() / sw.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
        vp, vl = predict(model, va_loader, dev)
        m = metrics(vp, vl)
        history.append({"epoch": epoch, **m})
        print(json.dumps({"epoch": epoch, "val_macro_f1": round(m["macro_f1"], 4)}), flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), args.out / "best.pt")

    model.load_state_dict(torch.load(args.out / "best.pt", map_location=dev))
    result = {
        "encoder": args.encoder,
        "input": args.input,
        "hp": hp,
        "seed": args.seed,
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "history": history,
    }
    for name, rows in (("val", va), ("test", te)):
        p, lab = predict(model, eval_loader(rows), dev)
        np.savez(
            args.out / f"{name}_probs.npz",
            ids=np.array([r["id"] for r in rows]),
            probs=p,
            labels=lab,
        )
        result[name] = metrics(p, lab)
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not args.keep_weights:
        (args.out / "best.pt").unlink()
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--input", choices=["jeju", "dual"], default="jeju")
    ap.add_argument("--data", type=Path, default=Path("data/v2"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="스모크용: train 앞 N개만")
    ap.add_argument("--keep-weights", action="store_true")
    for k, v in HP.items():
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(v), default=None)
    r = train(ap.parse_args(argv))
    print(json.dumps({k: r[k]["macro_f1"] for k in ("val", "test")}))


if __name__ == "__main__":
    main()

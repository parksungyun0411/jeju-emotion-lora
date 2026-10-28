"""성능 루프 분류 학습기 (Kaggle T4에서 실행).

  python -m jeju_lora.perf.train_cls --name r1_rl_pair --model klue/roberta-large \
      --revision <sha> --input pair --epochs 4 --lr 1.5e-5 --data-dir /tmp/data --out-dir out/r1

구조 (--arch)
  seqcls  AutoModelForSequenceClassification. 문장쌍이면 [CLS] 첫 문장 [SEP] 둘째 문장 [SEP]
  dual    Dual-Gated 인코더(원 Step 3): 공유 인코더로 제주어·표준어 [CLS]를 따로 뽑아
          g = σ(W[h_jeju; h_std])로 섞는다. 원 코드의 KR-BERT는 라이선스가 없어 다른 인코더를 쓴다
  llm     디코더 LLM + 분류 헤드(마지막 토큰), LoRA. --qlora면 기반 가중치를 4bit로 올린다

고르는 기준은 val macro-F1뿐이다. test는 라벨을 뺀 파일(test_nolabel.jsonl)만 받아 평가 시점마다
로짓을 저장하고 채점하지 않는다. 출력:
  out-dir/metrics.json  설정·평가 시점별 val 지표·최고 시점
  out-dir/logits.npz    val_logits[E,N,7] val_label val_row_id test_logits[E,M,7] test_row_id
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np

from jeju_lora.data import read_jsonl
from jeju_lora.labels import LABEL_NAMES, NUM_LABELS
from jeju_lora.perf.inputs import INPUT_MODES, TRAIN_MODES, llm_text, select_train, text_pair

LLM_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True, help="HF ID 또는 로컬 경로(DAPT 결과)")
    p.add_argument("--revision", default=None)
    p.add_argument("--arch", choices=["seqcls", "dual", "llm"], default="seqcls")
    p.add_argument("--input", choices=INPUT_MODES, default="pair")
    p.add_argument("--train-mode", choices=TRAIN_MODES, default="dedup")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=None, help="분류 헤드·게이트 학습률")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="none")
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--rdrop", type=float, default=0.0, help="R-Drop KL 계수 (0이면 끔)")
    p.add_argument("--llrd", type=float, default=1.0, help="층별 학습률 감쇠 (1이면 끔)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA 감쇠 (0이면 끔, 평가는 EMA 가중치)")
    p.add_argument("--hard-mining", type=float, default=0.0, help="직전 epoch 손실 상위 비율")
    p.add_argument("--hard-weight", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1, help="dual 헤드 드롭아웃")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evals-per-epoch", type=int, default=1)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--time-limit-min", type=float, default=0, help="넘으면 다음 평가 뒤 멈춘다")
    p.add_argument("--max-train", type=int, default=0, help="스모크용 train 표본 수")
    p.add_argument("--max-eval", type=int, default=0, help="스모크용 val·test 앞부분 수")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-model", action="store_true", help="최고 시점 가중치(LLM은 어댑터) 저장")
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(args) -> tuple[list[dict], list[dict], list[dict]]:
    d = Path(args.data_dir)
    train = select_train(read_jsonl(d / "train.jsonl"), args.train_mode)
    val = read_jsonl(d / "val.jsonl")
    test = read_jsonl(d / "test_nolabel.jsonl")
    if any("label" in r for r in test):
        raise SystemExit("test에 라벨이 있다 — 라벨을 뺀 test_nolabel.jsonl만 받는다")
    if args.max_train:
        train = random.Random(args.seed).sample(train, min(args.max_train, len(train)))
    if args.max_eval:
        val, test = val[: args.max_eval], test[: args.max_eval]
    return train, val, test


def class_weights(labels: list[int], kind: str) -> np.ndarray | None:
    if kind == "none":
        return None
    counts = np.maximum(np.bincount(labels, minlength=NUM_LABELS), 1).astype(float)
    w = counts.sum() / (NUM_LABELS * counts)
    if kind == "sqrt":
        w = np.sqrt(w)
    return w / w.mean()


def encode(tok, rows: list[dict], args, use_tti: bool) -> list[dict]:
    if args.arch == "llm":
        enc = tok([llm_text(r, args.input) for r in rows], truncation=True, max_length=args.max_len)
        items = [{"input_ids": ids} for ids in enc["input_ids"]]
    elif args.arch == "dual":
        ej = tok([r["jeju"] for r in rows], truncation=True, max_length=args.max_len)
        es = tok([r["std"] for r in rows], truncation=True, max_length=args.max_len)
        items = [
            {"jeju_ids": a, "std_ids": b}
            for a, b in zip(ej["input_ids"], es["input_ids"], strict=True)
        ]
    else:
        pairs = [text_pair(r, args.input) for r in rows]
        first = [a for a, _ in pairs]
        second = [b for _, b in pairs] if pairs[0][1] is not None else None
        enc = tok(
            first,
            second,
            truncation=True,
            max_length=args.max_len,
            return_token_type_ids=use_tti,
        )
        items = [{"input_ids": ids} for ids in enc["input_ids"]]
        if use_tti:
            for it, tt in zip(items, enc["token_type_ids"], strict=True):
                it["token_type_ids"] = tt
    for i, (it, r) in enumerate(zip(items, rows, strict=True)):
        it["label"] = int(r.get("label", -1))
        it["idx"] = i
    return items


def make_collate(pad_id: int):
    import torch

    def pad(seqs: list[list[int]], value: int):
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), value, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids, mask

    def collate(batch: list[dict]) -> dict:
        out = {
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        }
        for key, mask_key in (
            ("input_ids", "attention_mask"),
            ("jeju_ids", "jeju_mask"),
            ("std_ids", "std_mask"),
        ):
            if key in batch[0]:
                out[key], out[mask_key] = pad([b[key] for b in batch], pad_id)
        if "token_type_ids" in batch[0]:
            out["token_type_ids"] = pad([b["token_type_ids"] for b in batch], 0)[0]
        return out

    return collate


def build_dual(args):
    import torch
    from torch import nn
    from transformers import AutoModel

    class DualGated(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                args.model, revision=args.revision, dtype=torch.float32
            )
            hidden = self.encoder.config.hidden_size
            self.gate = nn.Linear(hidden * 2, hidden)
            self.dropout = nn.Dropout(args.dropout)
            self.classifier = nn.Linear(hidden, NUM_LABELS)

        def cls(self, ids, mask):
            return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

        def forward(self, jeju_ids, jeju_mask, std_ids, std_mask):
            h_j = self.cls(jeju_ids, jeju_mask)
            h_s = self.cls(std_ids, std_mask)
            g = torch.sigmoid(self.gate(torch.cat([h_j, h_s], dim=-1)))
            return self.classifier(self.dropout(g * h_j + (1 - g) * h_s))

    return DualGated()


def build_llm(args, tok):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification

    kw = {"num_labels": NUM_LABELS, "dtype": torch.float16 if args.fp16 else torch.float32}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kw["device_map"] = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, **kw
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(
        task_type="SEQ_CLS",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LLM_TARGETS,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lcfg)
    # fp16 기반 가중치 위에서 GradScaler를 쓰려면 학습 파라미터(LoRA·헤드)는 fp32여야 한다
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def load_tokenizer(args):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    except TypeError:
        # team-lucid DeBERTa-v3-ko: tokenizer.json은 WordPiece인데 transformers 5의
        # DebertaV2Tokenizer가 Unigram으로 다시 만들려다 실패한다 → tokenizer.json을 그대로 쓴다
        return PreTrainedTokenizerFast.from_pretrained(args.model, revision=args.revision)


def build_model(args):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from jeju_lora import compat

    compat.apply()  # kanana-1.5-2.1b 설정 검증 우회: 토크나이저를 부르기 전에 건다
    tok = load_tokenizer(args)
    use_tti = False
    if args.arch == "llm":
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        model = build_llm(args, tok)
    elif args.arch == "dual":
        model = build_dual(args)
    else:
        cfg = AutoConfig.from_pretrained(args.model, revision=args.revision, num_labels=NUM_LABELS)
        # KLUE RoBERTa·XLM-R는 type_vocab_size 1, DeBERTa-v3는 0: 둘째 문장 구분 id를 넣지 않는다
        use_tti = getattr(cfg, "type_vocab_size", 0) >= 2
        # transformers 5는 체크포인트 dtype(mDeBERTa는 fp16)으로 올린다 → fp32로 받아 학습
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, revision=args.revision, config=cfg, dtype=torch.float32
        )
    return tok, model, use_tti


def forward(model, batch: dict, arch: str):
    if arch == "dual":
        return model(batch["jeju_ids"], batch["jeju_mask"], batch["std_ids"], batch["std_mask"])
    kw = {k: batch[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch}
    return model(**kw).logits


def per_sample_loss(logits, labels, args, weight):
    import torch.nn.functional as F

    logits = logits.float()
    loss = F.cross_entropy(
        logits, labels, weight=weight, label_smoothing=args.label_smoothing, reduction="none"
    )
    if args.focal_gamma > 0:
        pt = (-F.cross_entropy(logits, labels, reduction="none")).exp()
        loss = (1 - pt) ** args.focal_gamma * loss
    return loss


def symmetric_kl(a, b):
    import torch.nn.functional as F

    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    return 0.5 * ((la.exp() * (la - lb)).sum(-1) + (lb.exp() * (lb - la)).sum(-1))


def param_groups(model, args) -> list[dict]:
    """가중치 감쇠 제외(bias·norm)와 층별 학습률 감쇠(LLRD). 헤드는 --head-lr."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    layer_ids = [int(m.group(1)) for n, _ in named if (m := re.search(r"\.layers?\.(\d+)\.", n))]
    n_layers = max(layer_ids) + 1 if layer_ids else 0
    groups: dict[tuple[float, float], list] = {}
    for n, p in named:
        no_decay = n.endswith("bias") or "norm" in n.lower()
        m = re.search(r"\.layers?\.(\d+)\.", n)
        if args.arch == "llm":
            is_head = "score" in n
            depth = n_layers
        elif m:
            is_head, depth = False, int(m.group(1)) + 1
        elif "embeddings" in n:
            is_head, depth = False, 0
        else:
            is_head, depth = True, n_layers + 1
        if is_head:
            lr = args.head_lr or args.lr
        else:
            lr = args.lr * args.llrd ** (n_layers - depth) if args.arch != "llm" else args.lr
        key = (lr, 0.0 if no_decay else args.weight_decay)
        groups.setdefault(key, []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


class EMA:
    """학습 파라미터의 지수이동평균. swap()을 두 번 부르면 원래 가중치로 돌아온다."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {
            n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    def swap(self, model) -> None:
        import torch

        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    cur = p.detach().float().clone()
                    p.copy_(self.shadow[n].to(p.dtype))
                    self.shadow[n] = cur


def macro_f1(y, pred) -> float:
    from sklearn.metrics import f1_score

    labels = list(range(NUM_LABELS))
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def per_class_f1(y, pred) -> dict:
    from sklearn.metrics import f1_score

    v = f1_score(y, pred, labels=list(range(NUM_LABELS)), average=None, zero_division=0)
    return {LABEL_NAMES[i]: round(float(x), 4) for i, x in enumerate(v)}


def eval_loader(items: list[dict], args, collate):
    """길이순으로 묶어 패딩을 줄이고, 결과는 원래 순서로 되돌린다."""
    from torch.utils.data import DataLoader

    def length(it):
        return sum(len(it[k]) for k in ("input_ids", "jeju_ids", "std_ids") if k in it)

    order = sorted(range(len(items)), key=lambda i: length(items[i]))
    loader = DataLoader([items[i] for i in order], batch_size=args.eval_batch, collate_fn=collate)
    return loader, np.argsort(order)


def predict(model, loader_order, args, device) -> np.ndarray:
    import torch

    loader, inverse = loader_order
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                outs.append(forward(model, batch, args.arch).float().cpu())
    model.train()
    return torch.cat(outs).numpy()[inverse]


def versions() -> dict:
    import torch
    import transformers

    out = {"torch": torch.__version__, "transformers": transformers.__version__}
    try:
        import peft

        out["peft"] = peft.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def train(args) -> dict:
    import torch
    from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    t0 = time.time()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train_rows, val_rows, test_rows = load_data(args)
    tok, model, use_tti = build_model(args)
    if not args.qlora:  # 4bit 모델은 불러올 때 device_map으로 이미 GPU에 있고 .to()를 막는다
        model.to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    collate = make_collate(pad_id)
    tr_items = encode(tok, train_rows, args, use_tti)
    va_items = encode(tok, val_rows, args, use_tti)
    te_items = encode(tok, test_rows, args, use_tti)
    va_loader = eval_loader(va_items, args, collate)
    te_loader = eval_loader(te_items, args, collate)
    y_val = np.array([r["label"] for r in val_rows])
    tr_labels = [it["label"] for it in tr_items]
    cw = class_weights(tr_labels, args.class_weight)
    weight = torch.tensor(cw, dtype=torch.float32, device=device) if cw is not None else None
    print(
        f"[{args.name}] train {len(tr_items)} val {len(va_items)} test {len(te_items)} "
        f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )

    opt = torch.optim.AdamW(param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    batches_per_epoch = math.ceil(len(tr_items) / args.batch)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = max(1, int(round(steps_per_epoch * args.epochs)))
    sched_fn = (
        get_cosine_schedule_with_warmup
        if args.scheduler == "cosine"
        else get_linear_schedule_with_warmup
    )
    sched = sched_fn(opt, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    gen = torch.Generator().manual_seed(args.seed)
    sample_w = np.ones(len(tr_items))
    last_loss = np.zeros(len(tr_items))

    history, val_logits, test_logits = [], [], []
    print(f"[{args.name}] loaded in {(time.time() - t0) / 60:.1f}m", flush=True)
    t_train, eval_sec = time.time(), 0.0
    step, epoch, stop = 0, 0, False
    model.train()
    while step < total_steps and not stop:
        if args.hard_mining > 0 and epoch > 0:
            # 원 Step 6: 직전 epoch 손실 상위 비율에 가중치를 주어 다시 뽑는다 (train 안에서만)
            k = int(len(last_loss) * args.hard_mining)
            sample_w = np.ones(len(tr_items))
            sample_w[np.argsort(-last_loss)[:k]] = args.hard_weight
            sampler = WeightedRandomSampler(
                sample_w.tolist(), len(tr_items), replacement=True, generator=gen
            )
        else:
            sampler = RandomSampler(tr_items, generator=gen)
        loader = DataLoader(tr_items, batch_size=args.batch, sampler=sampler, collate_fn=collate)
        run_loss, run_n = 0.0, 0
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                logits = forward(model, batch, args.arch)
                loss_vec = per_sample_loss(logits, batch["labels"], args, weight)
                if args.rdrop > 0:
                    logits2 = forward(model, batch, args.arch)
                    loss_vec = 0.5 * (
                        loss_vec + per_sample_loss(logits2, batch["labels"], args, weight)
                    )
                    loss_vec = loss_vec + args.rdrop * symmetric_kl(logits, logits2)
            last_loss[batch["idx"].cpu().numpy()] = loss_vec.detach().float().cpu().numpy()
            loss = loss_vec.mean() / args.grad_accum
            scaler.scale(loss).backward()
            run_loss += float(loss_vec.detach().mean())
            run_n += 1
            if (bi + 1) % args.grad_accum and bi + 1 != len(loader):
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if ema is not None:
                ema.update(model)
            step += 1
            if step % 100 == 0 or step == 20:
                el = time.time() - t_train - eval_sec
                print(
                    f"[{args.name}] step {step}/{total_steps} loss {run_loss / run_n:.4f} "
                    f"{el / step:.2f}s/step elapsed {(time.time() - t0) / 60:.1f}m",
                    flush=True,
                )
                run_loss, run_n = 0.0, 0
            if step % eval_every == 0 or step == total_steps:
                if ema is not None:
                    ema.swap(model)
                t_eval = time.time()
                vl = predict(model, va_loader, args, device)
                tl = predict(model, te_loader, args, device)
                eval_sec += time.time() - t_eval
                pred = vl.argmax(-1)
                f1 = macro_f1(y_val, pred)
                if args.save_model and all(f1 > h["val_f1_macro"] for h in history):
                    save_model(model, tok, args, out_dir)
                if ema is not None:
                    ema.swap(model)
                val_logits.append(vl)
                test_logits.append(tl)
                h = {
                    "step": step,
                    "epoch": round(step / steps_per_epoch, 3),
                    "val_f1_macro": f1,
                    "val_acc": float((pred == y_val).mean()),
                    "val_f1_per_class": per_class_f1(y_val, pred),
                    "elapsed_min": round((time.time() - t0) / 60, 2),
                    "eval_min_total": round(eval_sec / 60, 2),
                }
                history.append(h)
                print(f"[{args.name}] EVAL {json.dumps(h, ensure_ascii=False)}", flush=True)
                save_outputs(
                    out_dir, args, history, val_logits, test_logits, val_rows, test_rows, t0
                )
                if args.time_limit_min and (time.time() - t0) / 60 > args.time_limit_min:
                    print(f"[{args.name}] TIME_LIMIT {args.time_limit_min}m", flush=True)
                    stop = True
            if step >= total_steps or stop:
                break
        epoch += 1
    return save_outputs(out_dir, args, history, val_logits, test_logits, val_rows, test_rows, t0)


def save_model(model, tok, args, out_dir: Path) -> None:
    import torch

    target = out_dir / "model"
    target.mkdir(parents=True, exist_ok=True)
    if args.arch == "llm" or args.arch == "seqcls":
        model.save_pretrained(target)
        tok.save_pretrained(target)
    else:
        torch.save(model.state_dict(), target / "dual_state.pt")


def save_outputs(out_dir, args, history, val_logits, test_logits, val_rows, test_rows, t0) -> dict:
    best = int(np.argmax([h["val_f1_macro"] for h in history])) if history else -1
    summary = {
        "name": args.name,
        "args": vars(args),
        "versions": versions(),
        "n_evals": len(history),
        "best_index": best,
        "best": history[best] if history else None,
        "history": history,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if history:
        np.savez_compressed(
            out_dir / "logits.npz",
            val_logits=np.stack(val_logits).astype(np.float32),
            val_label=np.array([r["label"] for r in val_rows]),
            val_row_id=np.array([r["row_id"] for r in val_rows]),
            test_logits=np.stack(test_logits).astype(np.float32),
            test_row_id=np.array([r["row_id"] for r in test_rows]),
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train(args)
    print(f"[{args.name}] DONE best {json.dumps(summary['best'], ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()

"""PEFT LoRA + TRL SFTTrainer 학습.

데이터는 대화형 prompt-completion 형식이라 TRL이 completion(라벨 이름)에만 손실을 건다.
QLoRA 대신 bf16 LoRA를 쓴다: 4B 이하는 24GB에 올라가고, 병합 대상 가중치가
양자화 전 원본과 같아야 양자화 전후 비교가 깔끔하다.

  python -m jeju_lora.train_lora --config configs/lora_qwen3-4b.yaml
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from jeju_lora import compat
from jeju_lora.config import load_models, load_train_config
from jeju_lora.data import read_jsonl
from jeju_lora.prompts import to_sft_record


def model_dtype(tr) -> str:
    """기반 가중치 dtype. fp16 학습에서도 LoRA 어댑터는 PEFT가 fp32로 올린다."""
    if tr.bf16:
        return "bfloat16"
    return "float16" if tr.fp16 else "float32"


def build_datasets(cfg, train_path: str, val_path: str):
    from datasets import Dataset

    train = read_jsonl(train_path)
    val = read_jsonl(val_path)
    tr = cfg.train
    if tr.max_train_samples:
        train = random.Random(cfg.seed).sample(train, min(tr.max_train_samples, len(train)))
    if tr.max_eval_samples:
        val = random.Random(cfg.seed).sample(val, min(tr.max_eval_samples, len(val)))
    to_ds = lambda rows: Dataset.from_list([to_sft_record(r, cfg.input_mode) for r in rows])  # noqa: E731
    return to_ds(train), to_ds(val)


def make_trainer(cfg, model_id: str, revision: str | None, train_ds, eval_ds, **overrides):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tr, lo = cfg.train, cfg.lora
    args = SFTConfig(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        num_train_epochs=tr.epochs,
        learning_rate=tr.learning_rate,
        per_device_train_batch_size=tr.per_device_batch_size,
        per_device_eval_batch_size=tr.per_device_batch_size,
        gradient_accumulation_steps=tr.grad_accum,
        warmup_steps=tr.warmup_ratio,  # transformers 5: [0,1) 실수면 비율로 해석
        lr_scheduler_type=tr.lr_scheduler,
        max_length=tr.max_length,
        gradient_checkpointing=tr.gradient_checkpointing,
        bf16=tr.bf16,
        fp16=tr.fp16,
        logging_steps=tr.logging_steps,
        eval_strategy="steps",
        eval_steps=tr.eval_steps,
        save_strategy="steps",
        save_steps=tr.save_steps,
        save_total_limit=2,
        completion_only_loss=True,
        report_to="none",
        model_init_kwargs={"revision": revision, "dtype": model_dtype(tr)},
        **overrides,
    )
    peft_config = LoraConfig(
        r=lo.r,
        lora_alpha=lo.alpha,
        lora_dropout=lo.dropout,
        target_modules=lo.target_modules,
        task_type="CAUSAL_LM",
    )
    return SFTTrainer(
        model=model_id,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train", default="data/splits_sentence/train.jsonl")
    p.add_argument("--val", default="data/splits_sentence/val.jsonl")
    p.add_argument("--model-path", default=None, help="HF ID 대신 쓸 로컬 경로 (스모크용)")
    p.add_argument("--max-steps", type=int, default=None, help="스모크용 상한")
    args = p.parse_args()

    compat.apply()
    cfg = load_train_config(args.config)
    spec = load_models()[cfg.model]
    model_id = args.model_path or spec.hf_id
    revision = None if args.model_path else spec.revision

    train_ds, eval_ds = build_datasets(cfg, args.train, args.val)
    overrides = {"max_steps": args.max_steps} if args.max_steps else {}
    trainer = make_trainer(cfg, model_id, revision, train_ds, eval_ds, **overrides)
    result = trainer.train()
    metrics = {**result.metrics, **trainer.evaluate()}

    adapter_dir = Path(cfg.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    trainer.processing_class.save_pretrained(str(adapter_dir))
    (Path(cfg.output_dir) / "train_summary.json").write_text(
        json.dumps(
            {
                "model": cfg.model,
                "hf_id": model_id,
                "revision": revision,
                "config": asdict(cfg),
                "n_train": len(train_ds),
                "n_eval": len(eval_ds),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"adapter → {adapter_dir}")


if __name__ == "__main__":
    main()

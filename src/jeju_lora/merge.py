"""LoRA 어댑터를 기반 모델에 병합해 전체 가중치로 저장한다 (vLLM·AWQ 입력).
bf16이 기본이고, bf16이 없는 GPU(T4)에서는 --dtype float16을 쓴다.

  python -m jeju_lora.merge --model qwen3-4b --adapter outputs/lora/qwen3-4b/adapter \
      --out outputs/merged/qwen3-4b
"""

from __future__ import annotations

from pathlib import Path

from jeju_lora.config import load_models


def merge(base: str, revision: str | None, adapter: str, out: str, dtype: str = "bfloat16") -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, dtype=getattr(torch, dtype)
    )
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="configs/models.yaml의 키")
    p.add_argument("--model-path", default=None, help="HF ID 대신 쓸 로컬 경로 (스모크용)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = p.parse_args()

    spec = load_models()[args.model]
    base = args.model_path or spec.hf_id
    merge(base, None if args.model_path else spec.revision, args.adapter, args.out, args.dtype)
    print(f"merged → {args.out}")


if __name__ == "__main__":
    main()

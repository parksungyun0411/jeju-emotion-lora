"""병합 모델을 llm-compressor AWQ + W4A16_ASYM 프리셋으로 양자화한다
(4bit 가중치·16bit 활성값, 비대칭).

보정(calibration) 데이터는 학습 분할의 프롬프트+정답을 채팅 템플릿으로 만든 것이다.
평가 분할은 쓰지 않는다. 결과는 compressed-tensors 형식이라 vLLM이 그대로 읽는다.

--dtype float32는 fp16 보정이 실패하는 모델에 쓴다. Qwen3-4B 병합본을 fp16으로 보정하자
마지막 층 MLP 격자 탐색의 오차가 모두 inf가 되어 멈췄다. 활성값은 fp16 범위 안이었고,
제곱 오차 합이 fp16으로 계산되며 넘친 것으로 보인다(reports/diagnostics/qwen3_fp16.md).
bf16이 없는 T4에서는 fp32로 CPU에 올리고 llm-compressor가 층 단위로 GPU에 올려 보정한다.

  python -m jeju_lora.quantize --model-dir outputs/merged/qwen3-4b --out outputs/awq/qwen3-4b
"""

from __future__ import annotations

import random

from jeju_lora import compat
from jeju_lora.data import read_jsonl
from jeju_lora.labels import EMOTION_LABELS
from jeju_lora.prompts import build_messages

SCHEME = "W4A16_ASYM"
DTYPES = ("auto", "float16", "bfloat16", "float32")


def resolve_dtype(torch, name: str):
    """'auto'는 체크포인트의 dtype을 그대로 쓴다."""
    if name not in DTYPES:
        raise ValueError(f"dtype은 {DTYPES} 중 하나")
    return name if name == "auto" else getattr(torch, name)


def calibration_texts(
    tokenizer, train: list[dict], n: int, input_mode: str, seed: int
) -> list[str]:
    rows = random.Random(seed).sample(train, min(n, len(train)))
    texts = []
    for r in rows:
        messages = build_messages(r, input_mode) + [
            {"role": "assistant", "content": EMOTION_LABELS[int(r["label"])]}
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return texts


def build_recipe():
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    return [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme=SCHEME, targets=["Linear"]),
    ]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train", default="data/splits_sentence/train.jsonl")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--input-mode", default="jeju")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=DTYPES, default="auto", help="보정 때 모델을 올릴 dtype")
    args = p.parse_args()

    compat.apply()
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=resolve_dtype(torch, args.dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    texts = calibration_texts(
        tokenizer, read_jsonl(args.train), args.num_samples, args.input_mode, args.seed
    )
    ds = Dataset.from_dict({"text": texts}).map(
        lambda s: tokenizer(
            s["text"],
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        ),
        remove_columns=["text"],
    )
    oneshot(
        model=model,
        dataset=ds,
        recipe=build_recipe(),
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(texts),
    )
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)
    print(f"AWQ {SCHEME} → {args.out}")


if __name__ == "__main__":
    main()

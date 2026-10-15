"""병합 모델을 llm-compressor AWQ + W4A16_ASYM 프리셋으로 양자화한다
(4bit 가중치·16bit 활성값, 비대칭).

보정(calibration) 데이터는 학습 분할의 프롬프트+정답을 채팅 템플릿으로 만든 것이다.
평가 분할은 쓰지 않는다. 결과는 compressed-tensors 형식이라 vLLM이 그대로 읽는다.

  python -m jeju_lora.quantize --model-dir outputs/merged/qwen3-4b --out outputs/awq/qwen3-4b
"""

from __future__ import annotations

import random

from jeju_lora import compat
from jeju_lora.data import read_jsonl
from jeju_lora.labels import EMOTION_LABELS
from jeju_lora.prompts import build_messages

SCHEME = "W4A16_ASYM"


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
    args = p.parse_args()

    compat.apply()
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype="auto")
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

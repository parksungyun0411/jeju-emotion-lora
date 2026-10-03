"""프롬프트 구성. 제로샷·퓨샷·LoRA 학습·평가가 모두 같은 함수를 쓴다."""

from __future__ import annotations

import random

from jeju_lora.labels import EMOTION_LABELS, LABEL_NAMES, NUM_LABELS

SYSTEM_PROMPT = (
    "너는 제주어 문장의 감정을 분류한다. "
    f"다음 {NUM_LABELS}개 중 하나만 답한다: {', '.join(LABEL_NAMES)}. "
    "설명 없이 감정 이름 하나만 쓴다."
)

INPUT_MODES = ("jeju", "jeju+std")


def user_message(item: dict, input_mode: str = "jeju") -> str:
    if input_mode == "jeju":
        return f"제주어: {item['jeju']}\n감정:"
    if input_mode == "jeju+std":
        return f"제주어: {item['jeju']}\n표준어: {item['std']}\n감정:"
    raise ValueError(f"input_mode는 {INPUT_MODES} 중 하나: {input_mode}")


def pick_few_shot(train: list[dict], per_class: int, seed: int = 42) -> list[dict]:
    """클래스마다 per_class개씩 뽑아 클래스가 섞이도록 돌려가며 배치한다."""
    if per_class <= 0:
        return []
    rng = random.Random(seed)
    by_class: dict[int, list[dict]] = {i: [] for i in range(NUM_LABELS)}
    for item in train:
        by_class[int(item["label"])].append(item)
    picked = {i: rng.sample(items, min(per_class, len(items))) for i, items in by_class.items()}
    ordered = []
    for k in range(per_class):
        for i in range(NUM_LABELS):
            if k < len(picked[i]):
                ordered.append(picked[i][k])
    return ordered


def build_messages(
    item: dict, input_mode: str = "jeju", shots: list[dict] | None = None
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots or []:
        messages.append({"role": "user", "content": user_message(shot, input_mode)})
        messages.append({"role": "assistant", "content": EMOTION_LABELS[int(shot["label"])]})
    messages.append({"role": "user", "content": user_message(item, input_mode)})
    return messages


def to_sft_record(item: dict, input_mode: str = "jeju") -> dict:
    """TRL 대화형 prompt-completion 형식. 손실은 completion(라벨 이름)에만 걸린다."""
    return {
        "prompt": build_messages(item, input_mode),
        "completion": [{"role": "assistant", "content": EMOTION_LABELS[int(item["label"])]}],
    }

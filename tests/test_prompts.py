from collections import Counter

import pytest

from jeju_lora.labels import LABEL_NAMES
from jeju_lora.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    pick_few_shot,
    to_sft_record,
    user_message,
)

ITEM = {"row_id": 1, "jeju": "혼저 옵서예", "std": "어서 오세요", "label": 2}


def test_system_prompt_lists_every_label():
    for name in LABEL_NAMES:
        assert name in SYSTEM_PROMPT


def test_user_message_modes():
    assert user_message(ITEM) == "제주어: 혼저 옵서예\n감정:"
    assert "표준어: 어서 오세요" in user_message(ITEM, "jeju+std")
    with pytest.raises(ValueError):
        user_message(ITEM, "std")


def test_zero_shot_messages():
    msgs = build_messages(ITEM)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "슬픔" not in msgs[-1]["content"]  # 정답이 프롬프트에 새지 않는다


def test_few_shot_balanced_deterministic(items):
    shots = pick_few_shot(items, 2, seed=1)
    assert len(shots) == 14
    assert Counter(s["label"] for s in shots) == {c: 2 for c in range(7)}
    assert [s["label"] for s in shots[:7]] == list(range(7))  # 클래스가 돌아가며 나온다
    assert shots == pick_few_shot(items, 2, seed=1)
    assert pick_few_shot(items, 0) == []
    msgs = build_messages(ITEM, shots=shots)
    assert len(msgs) == 1 + 2 * 14 + 1
    assert msgs[2] == {"role": "assistant", "content": LABEL_NAMES[shots[0]["label"]]}


def test_sft_record():
    rec = to_sft_record(ITEM)
    assert rec["completion"] == [{"role": "assistant", "content": "슬픔"}]
    assert rec["prompt"] == build_messages(ITEM)

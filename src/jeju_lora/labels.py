"""감정 라벨 정의와 생성문 → 라벨 파싱.

라벨 번호·이름은 원 프로젝트(jeju-emotion-analysis/src/jeju_kr_bert.py)의
EMOTION_LABELS를 그대로 쓴다. README에는 '기쁨'으로 적혀 있지만 코드와 데이터는 '행복'이다.
"""

from __future__ import annotations

import re

EMOTION_LABELS: dict[int, str] = {
    0: "중립",
    1: "슬픔",
    2: "행복",
    3: "분노",
    4: "놀람",
    5: "공포",
    6: "혐오",
}
NUM_LABELS = len(EMOTION_LABELS)
LABEL_NAMES: list[str] = [EMOTION_LABELS[i] for i in range(NUM_LABELS)]
NAME_TO_ID: dict[str, int] = {name: i for i, name in EMOTION_LABELS.items()}

# 모델이 자주 내놓는 동의어. 값은 라벨 번호.
SYNONYMS: dict[str, int] = {
    "기쁨": 2,
    "즐거움": 2,
    "화남": 3,
    "놀라움": 4,
    "두려움": 5,
    "무서움": 5,
    "역겨움": 6,
    "neutral": 0,
    "sadness": 1,
    "happiness": 2,
    "joy": 2,
    "anger": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
}

INVALID = -1

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_label(text: str | None) -> int:
    """생성문에서 라벨 번호를 꺼낸다. 찾지 못하면 INVALID(-1).

    규칙: <think> 블록 제거 → 라벨명·동의어 중 가장 앞에 나온 것을 고른다.
    라벨명이 없고 0~6 숫자 하나로만 답했으면 그 숫자를 쓴다.
    """
    if not text:
        return INVALID
    s = _THINK_RE.sub("", text).strip()
    if not s:
        return INVALID
    lowered = s.lower()
    best_pos, best_id = None, INVALID
    for name, idx in list(NAME_TO_ID.items()) + list(SYNONYMS.items()):
        pos = lowered.find(name)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos, best_id = pos, idx
    if best_id != INVALID:
        return best_id
    m = re.fullmatch(r"\s*([0-6])\s*[.)]?\s*", s)
    if m:
        return int(m.group(1))
    return INVALID

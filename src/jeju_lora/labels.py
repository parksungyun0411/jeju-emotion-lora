"""감정 라벨 정의와 생성문 → 라벨 파싱.

번호 → 이름은 데이터에서 정했다. 원본(merged_jeju_balanced_sorted.xlsx)의 감정번호 × gpt감정
교차표는 대각선만 채워져 있다: 0 중립 18,000 · 1 행복 17,115 · 2 슬픔 17,195 · 3 분노 17,840 ·
4 놀람 18,612 · 5 공포 20,430 · 6 혐오 18,132 (전처리 전 행 수). 문장 내용도 1은 긍정,
2는 부정이다. 원 저장소 코드(재현 코드 jeju_kr_bert.py와 라벨링 스크립트 automation.py 등)는
2026-10-28(jeju_kr_bert.py는 04f61bd, automation.py 등은 6bc5b7f) 이전 판에서 1=슬픔, 2=행복으로
1·2가 뒤바뀌어 있었고, 이 저장소도 처음에 그것을 베꼈다. 원 README의 감정 순서(중립·행복·슬픔…)는
데이터와 맞는다. tests/test_labels.py가 이 대응을 고정하고, 원본이 있으면 교차표로 다시 확인한다.
"""

from __future__ import annotations

import re

EMOTION_LABELS: dict[int, str] = {
    0: "중립",
    1: "행복",
    2: "슬픔",
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
    "기쁨": 1,
    "즐거움": 1,
    "화남": 3,
    "놀라움": 4,
    "두려움": 5,
    "무서움": 5,
    "역겨움": 6,
    "neutral": 0,
    "happiness": 1,
    "joy": 1,
    "sadness": 2,
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

"""성능 루프 입력 구성. torch 없이 돌아가는 부분만 둔다(단위 테스트 대상).

GPT-4o 라벨은 원 automation.py가 '표준어 문장' 열을 넣어 붙였다. 그래서 표준어를 앞에 두는
문장쌍(pair)을 기본으로 보고, 제주어만(jeju)·표준어만(std)·순서를 바꾼 쌍(pair_jeju)을 비교한다.
원 Dual-Gated KR-BERT도 val·test에서 표준어 열을 입력으로 쓴다.
"""

from __future__ import annotations

import re

INPUT_MODES = ("jeju", "std", "pair", "pair_jeju")
TRAIN_MODES = ("dedup", "dup")

LLM_INSTRUCTION = "다음 문장의 감정을 중립, 행복, 슬픔, 분노, 놀람, 공포, 혐오 중 하나로 분류한다."


def dedup_train(rows: list[dict]) -> list[dict]:
    """같은 제주어 문장의 복제 행(원 파일의 클래스 균형용)을 첫 행 하나로 줄인다.

    문장 단위 분할은 라벨이 둘 이상인 문장을 이미 뺐으므로 복제 행의 라벨은 모두 같다.
    """
    seen: dict[str, dict] = {}
    for r in rows:
        prev = seen.setdefault(r["jeju"], r)
        if prev["label"] != r["label"]:
            raise ValueError(f"같은 문장에 라벨이 둘: row_id {prev['row_id']}, {r['row_id']}")
    return list(seen.values())


def select_train(rows: list[dict], mode: str) -> list[dict]:
    if mode == "dedup":
        return dedup_train(rows)
    if mode == "dup":
        return list(rows)
    raise ValueError(f"train_mode는 {TRAIN_MODES} 중 하나: {mode}")


def text_pair(row: dict, mode: str) -> tuple[str, str | None]:
    """인코더 입력 (첫 문장, 둘째 문장). 둘째가 None이면 단일 문장 입력."""
    if mode == "jeju":
        return row["jeju"], None
    if mode == "std":
        return row["std"], None
    if mode == "pair":
        return row["std"], row["jeju"]
    if mode == "pair_jeju":
        return row["jeju"], row["std"]
    raise ValueError(f"input_mode는 {INPUT_MODES} 중 하나: {mode}")


def llm_text(row: dict, mode: str) -> str:
    """디코더 LLM 분류 헤드 입력. 마지막 토큰의 은닉 상태로 분류하므로 '감정:'으로 끝낸다."""
    lines = [LLM_INSTRUCTION]
    if mode in ("std", "pair"):
        lines.append(f"표준어: {row['std']}")
    if mode in ("jeju", "pair", "pair_jeju"):
        lines.append(f"제주어: {row['jeju']}")
    if mode == "pair_jeju":
        lines.append(f"표준어: {row['std']}")
    if mode not in INPUT_MODES:
        raise ValueError(f"input_mode는 {INPUT_MODES} 중 하나: {mode}")
    lines.append("감정:")
    return "\n".join(lines)


def strip_labels(rows: list[dict]) -> list[dict]:
    """test를 커널에 올릴 때 라벨을 뺀다. 커널 안에서는 test 지표를 계산할 수 없게 한다."""
    return [{"row_id": r["row_id"], "jeju": r["jeju"], "std": r["std"]} for r in rows]


def normalize_ws(text: str) -> str:
    """중복 제거 비교용: 공백을 하나로 줄이고 양끝을 자른다."""
    return re.sub(r"\s+", " ", str(text)).strip()

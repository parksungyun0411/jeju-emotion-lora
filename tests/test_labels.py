from pathlib import Path

import pytest

from jeju_lora.data import DEFAULT_XLSX, LABEL_COL, read_table
from jeju_lora.labels import (
    EMOTION_LABELS,
    INVALID,
    LABEL_NAMES,
    NAME_TO_ID,
    SYNONYMS,
    parse_label,
)

RAW_XLSX = Path(__file__).resolve().parents[1] / DEFAULT_XLSX


def test_label_mapping_pinned():
    # 원본 데이터의 감정번호 × gpt감정 교차표로 정한 대응 (labels.py docstring)
    assert EMOTION_LABELS == {
        0: "중립",
        1: "행복",
        2: "슬픔",
        3: "분노",
        4: "놀람",
        5: "공포",
        6: "혐오",
    }
    assert LABEL_NAMES == ["중립", "행복", "슬픔", "분노", "놀람", "공포", "혐오"]


@pytest.mark.parametrize(
    "word,name",
    [
        ("기쁨", "행복"),
        ("즐거움", "행복"),
        ("happiness", "행복"),
        ("joy", "행복"),
        ("sadness", "슬픔"),
        ("화남", "분노"),
        ("anger", "분노"),
        ("놀라움", "놀람"),
        ("surprise", "놀람"),
        ("두려움", "공포"),
        ("무서움", "공포"),
        ("fear", "공포"),
        ("역겨움", "혐오"),
        ("disgust", "혐오"),
        ("neutral", "중립"),
    ],
)
def test_synonyms_point_to_named_label(word, name):
    assert SYNONYMS[word] == NAME_TO_ID[name]


def test_every_synonym_checked():
    assert len(SYNONYMS) == 15


@pytest.mark.skipif(not RAW_XLSX.exists(), reason="원본 xlsx가 없다 (AI Hub 파생, 저장소에 없음)")
def test_mapping_matches_raw_crosstab():
    import pandas as pd

    df = read_table(RAW_XLSX)
    table = pd.crosstab(df[LABEL_COL], df["gpt감정"])
    assert sorted(table.index) == list(range(len(EMOTION_LABELS)))
    for i in table.index:
        assert table.loc[i].idxmax() == EMOTION_LABELS[int(i)]


@pytest.mark.parametrize("i,name", list(enumerate(LABEL_NAMES)))
def test_exact_label_names(i, name):
    assert parse_label(name) == i


@pytest.mark.parametrize(
    "text,expected",
    [
        ("  분노\n", 3),
        ("감정: 슬픔", 2),
        ("감정: 행복", 1),
        ("이 문장의 감정은 '공포'입니다.", 5),
        ("기쁨", 1),  # 동의어 → 행복
        ("Sadness", 2),
        ("Disgust", 6),
        ("<think>중립일까 분노일까</think>놀람", 4),  # 사고 블록 무시
        ("슬픔 아니면 분노", 2),  # 가장 앞에 나온 라벨
        ("행복 아니면 슬픔", 1),
        ("3", 3),
        ("3.", 3),
        ("", INVALID),
        (None, INVALID),
        ("모르겠다", INVALID),
        ("7", INVALID),
        ("12", INVALID),
        ("<think>분노</think>", INVALID),
    ],
)
def test_parse_cases(text, expected):
    assert parse_label(text) == expected

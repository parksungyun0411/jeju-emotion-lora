import pytest

from jeju_lora.labels import INVALID, LABEL_NAMES, parse_label


@pytest.mark.parametrize("i,name", list(enumerate(LABEL_NAMES)))
def test_exact_label_names(i, name):
    assert parse_label(name) == i


@pytest.mark.parametrize(
    "text,expected",
    [
        ("  분노\n", 3),
        ("감정: 슬픔", 1),
        ("이 문장의 감정은 '공포'입니다.", 5),
        ("기쁨", 2),  # 동의어 → 행복
        ("Disgust", 6),
        ("<think>중립일까 분노일까</think>놀람", 4),  # 사고 블록 무시
        ("슬픔 아니면 분노", 1),  # 가장 앞에 나온 라벨
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

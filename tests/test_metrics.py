import pytest

from jeju_lora.metrics import classification_metrics, cohen_kappa


def test_macro_f1_hand_computed():
    # class0: TP1 FP0 FN1 → F1 2/3
    # class1: TP2 FP1 FN0 → P 2/3, R 1 → F1 0.8
    # class2: TP1 FP0 FN1 (마지막은 파싱 실패) → F1 2/3
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, -1]
    m = classification_metrics(y_true, y_pred)
    assert m["f1_macro"] == pytest.approx((2 / 3 + 0.8 + 2 / 3) / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["invalid_rate"] == pytest.approx(1 / 6)
    assert m["f1_per_class"] == pytest.approx({"중립": 2 / 3, "슬픔": 0.8, "행복": 2 / 3})
    # weighted: 지지도 2·2·2 → macro와 같다
    assert m["f1_weighted"] == pytest.approx(m["f1_macro"])
    assert m["confusion"][2][2] == 1 and sum(m["confusion"][2]) == 1  # -1은 행렬에서 빠진다


def test_macro_f1_counts_wrong_class_predictions():
    # 정답에 없는 클래스(3)를 예측하면 그 클래스 F1=0이 평균에 들어간다
    # class0: TP1 FN1 → 2/3 ; class3: TP0 FP1 → 0
    m = classification_metrics([0, 0], [0, 3])
    assert m["f1_macro"] == pytest.approx((2 / 3 + 0) / 2)


def test_perfect_and_all_invalid():
    assert classification_metrics([0, 1, 6], [0, 1, 6])["f1_macro"] == 1.0
    m = classification_metrics([3, 4], [-1, -1])
    assert m["f1_macro"] == 0.0 and m["invalid_rate"] == 1.0


def test_metrics_input_errors():
    with pytest.raises(ValueError):
        classification_metrics([0], [0, 1])
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_kappa_hand_computed():
    # po=3/4, pe=0.5·0.25+0.5·0.75=0.5 → (0.75-0.5)/0.5 = 0.5
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)
    assert cohen_kappa([2, 3, 4], [2, 3, 4]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1], [1, 1]) == 1.0
    # 완전 반대: po=0, pe=0.5 → -1
    assert cohen_kappa([0, 1], [1, 0]) == pytest.approx(-1.0)

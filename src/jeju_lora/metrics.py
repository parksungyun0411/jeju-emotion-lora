"""분류 지표. 파싱 실패(-1)는 어느 클래스로도 맞히지 못한 예측으로 센다."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from jeju_lora.labels import INVALID, LABEL_NAMES, NUM_LABELS

CLASSES = list(range(NUM_LABELS))


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true와 y_pred 길이가 다르다")
    if len(y_true) == 0:
        raise ValueError("빈 입력")
    # 평균 대상은 정답·예측에 나온 클래스(원 프로젝트의 sklearn 기본 동작과 같다).
    # 단 -1(파싱 실패)은 클래스로 세지 않아 정답 클래스의 FN으로만 잡힌다.
    present = sorted((set(y_true.tolist()) | set(y_pred.tolist())) - {INVALID})
    per_class = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)
        ),
        "f1_per_class": {LABEL_NAMES[c]: float(v) for c, v in zip(present, per_class, strict=True)},
        "invalid_rate": float(np.mean(y_pred == INVALID)),
        "confusion": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """두 평정자 간 Cohen's kappa. 기대 일치도가 1이면(둘 다 한 클래스만) 1.0을 돌려준다."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("길이가 같고 비어 있지 않아야 한다")
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)

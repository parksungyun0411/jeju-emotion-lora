"""합성 데이터 픽스처. 실제 데이터(AI Hub 기반 라벨링 파일)는 저장소에 없다."""

import numpy as np
import pandas as pd
import pytest

from jeju_lora.data import JEJU_COL, LABEL_COL, STD_COL


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """원 파일과 같은 열 구성. 클래스당 60행 + 결측 행 3개."""
    rows = []
    for i in range(7 * 60):
        rows.append({JEJU_COL: f"제주 문장 {i}", STD_COL: f"표준 문장 {i}", LABEL_COL: i % 7})
    rows.insert(5, {JEJU_COL: None, STD_COL: "x", LABEL_COL: 1})
    rows.insert(50, {JEJU_COL: "y", STD_COL: None, LABEL_COL: 2})
    rows.insert(100, {JEJU_COL: "z", STD_COL: "z", LABEL_COL: np.nan})
    return pd.DataFrame(rows)


def make_items(n_per_class: int = 20) -> list[dict]:
    return [
        {"row_id": i, "jeju": f"제주 {i}", "std": f"표준 {i}", "label": i % 7}
        for i in range(7 * n_per_class)
    ]


@pytest.fixture
def items() -> list[dict]:
    return make_items()

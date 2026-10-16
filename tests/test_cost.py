import json

import pytest

from jeju_lora.cost import PENDING, build_rows, render_markdown, usd_per_million

CFG = {
    "rows": [
        {"name": "A", "metrics": "a", "throughput": "a"},
        {"name": "B zero-shot", "metrics": "b"},
    ]
}


def test_usd_per_million_hand_computed():
    # $0.69/h, 100건/s → 36만 건/h → 0.69 / 360000 × 1e6 = 1.91666…
    assert usd_per_million(0.69, 100) == pytest.approx(0.69 / 360000 * 1e6)
    assert usd_per_million(0.69, 100) == pytest.approx(1.916666, rel=1e-5)
    # 1건/s, $3.6/h → 3600건에 3.6달러 → 100만 건에 1000달러
    assert usd_per_million(3.6, 1) == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        usd_per_million(1.0, 0)
    with pytest.raises(ValueError):
        usd_per_million(-1.0, 10)


def test_all_pending_when_nothing_measured(tmp_path):
    rows = build_rows(CFG, tmp_path, 0.5)
    md = render_markdown(rows, None, "")
    assert md.count(PENDING) >= 8
    line_b = next(line for line in md.splitlines() if line.startswith("| B"))
    assert line_b.endswith("| — | — | — |")  # 처리량 측정 대상이 아닌 행


def test_rows_from_files(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "throughput").mkdir()
    (tmp_path / "metrics" / "a.json").write_text(
        json.dumps(
            {
                "machine_label": {"f1_macro": 0.5, "invalid_rate": 0.0},
                "human_label": None,
            }
        )
    )
    (tmp_path / "throughput" / "a.json").write_text(
        json.dumps({"items_per_s": 200.0, "latency_p95_s": 0.25})
    )
    rows = build_rows(CFG, tmp_path, 0.72)
    a = rows[0]
    assert a["usd_per_million"] == pytest.approx(0.72 / (200 * 3600) * 1e6)
    assert a["p95_ms"] == pytest.approx(250)
    assert a["f1_human"] is None
    md = render_markdown(rows, 0.72, "test-gpu")
    assert (
        "| A | 측정 전 | 측정 전 | 측정 전 | 0.5000 | 측정 전 | 0.0000 | 200.0 | 250 | 1.00 |" in md
    )


def test_encoder_row_without_p95(tmp_path):
    (tmp_path / "throughput").mkdir()
    (tmp_path / "throughput" / "a.json").write_text(
        json.dumps({"items_per_s": 1000.0, "latency_p95_s": None, "system": "encoder"})
    )
    rows = build_rows(CFG, tmp_path, 3.6)
    md = render_markdown(rows, 3.6, "g")
    assert (
        "| A | 측정 전 | 측정 전 | PyTorch 배치 | 측정 전 | 측정 전 | 측정 전 | 1000.0 | — | 1.00 |"
        in md
    )


def test_split_precision_backend_come_from_files(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "throughput").mkdir()
    machine = {"f1_macro": 0.6, "invalid_rate": 0.01}
    (tmp_path / "metrics" / "a.json").write_text(
        json.dumps(
            {
                "backend": "openai",
                "constrained": True,
                "split": "splits_sentence",
                "precision": "fp16",
                "machine_label": machine,
            }
        )
    )
    (tmp_path / "throughput" / "a.json").write_text(
        json.dumps({"items_per_s": 10.0, "latency_p95_s": 1.5, "precision": "awq-w4a16"})
    )
    (tmp_path / "metrics" / "b.json").write_text(
        json.dumps(
            {"backend": "openai", "constrained": False, "split": "splits", "machine_label": machine}
        )
    )
    rows = build_rows(CFG, tmp_path, None)
    # 정밀도는 지표 파일 값이 먼저다
    assert (rows[0]["split"], rows[0]["precision"], rows[0]["backend"]) == (
        "문장 단위",
        "fp16",
        "vLLM",
    )
    assert (rows[1]["split"], rows[1]["backend"]) == ("원 행 단위", "vLLM (제약 없음)")
    assert rows[0]["usd_per_million"] is None


def test_not_run_rows_show_reason(tmp_path):
    cfg = {"rows": [{"name": "C LoRA", "metrics": "c", "not_run": "T4 할당 한도"}]}
    rows = build_rows(cfg, tmp_path, 0.1)
    md = render_markdown(rows, 0.1, "T4", price_note="출처 X")
    assert "| C LoRA |" + " 측정 안 함 |" * 9 in md
    assert "- C LoRA: T4 할당 한도" in md
    assert "요금 근거: 출처 X" in md


def test_no_human_rows_show_dash(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "r.json").write_text(
        json.dumps({"split": "splits", "machine_label": {"f1_macro": 0.8, "invalid_rate": 0.0}})
    )
    cfg = {"rows": [{"name": "R", "metrics": "r", "no_human": True}]}
    md = render_markdown(build_rows(cfg, tmp_path, None), None, "g")
    assert "| R | 원 행 단위 | 측정 전 | 측정 전 | 0.8000 | — | 0.0000 | — | — | — |" in md

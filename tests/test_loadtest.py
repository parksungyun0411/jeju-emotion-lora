import asyncio
import functools

import httpx
import pytest

from jeju_lora import loadtest
from jeju_lora.bench_encoder import ensemble_items_per_s
from jeju_lora.generation import OpenAIChatBackend
from jeju_lora.quantize import calibration_texts


def test_percentile_nearest_rank():
    v = [float(i) for i in range(1, 101)]  # 1..100
    assert loadtest.percentile(v, 95) == 95.0
    assert loadtest.percentile(v, 50) == 50.0
    assert loadtest.percentile(v, 100) == 100.0
    assert loadtest.percentile([3.0, 1.0, 2.0], 50) == 2.0
    with pytest.raises(ValueError):
        loadtest.percentile([], 95)


def test_summarize():
    s = loadtest.summarize([0.1] * 10, wall_s=2.0, errors=1)
    assert s["items_per_s"] == 5.0 and s["errors"] == 1 and s["latency_p95_s"] == 0.1


def test_run_against_mock_server(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 7:
            return httpx.Response(500)
        return httpx.Response(200, json={"choices": [{"message": {"content": "중립"}}]})

    monkeypatch.setattr(
        loadtest.httpx,
        "AsyncClient",
        functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
    )
    backend = OpenAIChatBackend("http://mock/v1", "m")
    msgs = [[{"role": "user", "content": str(i)}] for i in range(25)]
    res = asyncio.run(loadtest.run(backend, msgs, concurrency=4, warmup=5))
    assert len(seen) == 25
    assert res["completed"] + res["errors"] == 20
    assert res["errors"] == 1  # 7번째 요청은 측정 구간(6번째 이후)
    assert res["items_per_s"] > 0


def test_ensemble_throughput_hand_computed():
    # KoELECTRA 1회 @1000/s + KR-BERT 2회 @800/s → 1/(0.001 + 0.0025) = 285.71…
    assert ensemble_items_per_s([(1, 1000.0), (2, 800.0)]) == pytest.approx(1 / 0.0035)
    with pytest.raises(ValueError):
        ensemble_items_per_s([(1, 0.0)])


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_calibration_texts_use_train_with_answer(items):
    texts = calibration_texts(StubTokenizer(), items, 10, "jeju", seed=0)
    assert len(texts) == 10
    assert all(t.startswith("system:") for t in texts)
    assert all(
        t.rsplit("assistant:", 1)[1] in {"중립", "행복", "슬픔", "분노", "놀람", "공포", "혐오"}
        for t in texts
    )
    assert texts == calibration_texts(StubTokenizer(), items, 10, "jeju", seed=0)

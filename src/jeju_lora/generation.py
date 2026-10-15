"""생성 백엔드. 모두 `generate(list[messages]) -> list[str]`를 제공한다.

- OpenAIChatBackend: vLLM `vllm serve`의 OpenAI 호환 /v1/chat/completions.
  constrained=True면 structured_outputs.choice로 출력을 라벨 7개 중 하나로 묶는다
  (vLLM v0.12부터 guided_choice 대신 이 필드를 쓴다).
- HFBackend: transformers generate. vLLM 없이 LoRA 어댑터를 바로 확인할 때 쓴다.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

import httpx

from jeju_lora.labels import LABEL_NAMES


def auto_dtype(torch, device: str):
    """bf16을 지원하는 GPU는 bf16, T4 같은 GPU는 fp16, CPU는 fp32."""
    if device != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Backend(Protocol):
    def generate(self, batch: list[list[dict]]) -> list[str]: ...


class OpenAIChatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        constrained: bool = True,
        max_tokens: int = 8,
        concurrency: int = 32,
        timeout: float = 120.0,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.constrained = constrained
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.timeout = timeout
        self.latencies: list[float] = []

    def payload(self, messages: list[dict]) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.constrained:
            body["structured_outputs"] = {"choice": LABEL_NAMES}
        return body

    async def _one(self, client: httpx.AsyncClient, sem: asyncio.Semaphore, messages):
        async with sem:
            t0 = time.perf_counter()
            r = await client.post(self.url, json=self.payload(messages))
            r.raise_for_status()
            self.latencies.append(time.perf_counter() - t0)
            return r.json()["choices"][0]["message"]["content"] or ""

    async def agenerate(self, batch: list[list[dict]]) -> list[str]:
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await asyncio.gather(*(self._one(client, sem, m) for m in batch))

    def generate(self, batch: list[list[dict]]) -> list[str]:
        return asyncio.run(self.agenerate(batch))


class HFBackend:
    def __init__(
        self,
        model_path: str,
        adapter: str | None = None,
        revision: str | None = None,
        batch_size: int = 32,
        max_new_tokens: int = 8,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from jeju_lora import compat

        compat.apply()
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = auto_dtype(torch, device)
        model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, dtype=dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def generate(self, batch: list[list[dict]]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(batch), self.batch_size):
            chunk = batch[i : i + self.batch_size]
            texts = [
                self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in chunk
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return out

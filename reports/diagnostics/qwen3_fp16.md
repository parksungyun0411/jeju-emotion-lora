# Qwen3-4B fp16 진단

2026-10-16 Kaggle T4에서 Qwen3-4B LoRA 병합본의 AWQ 양자화가 fp16 보정 도중 멈췄다
(`Exception: No finite loss was found in best scalesgrid search`). 마지막 층(36층 중 35번)
`post_attention_layernorm → MLP` 격자 탐색 20개 격자점의 오차가 모두 `inf`였다.
같은 병합본을 fp16으로 서빙해 잰 macro-F1 0.4738도 넘침의 영향을 받았는지 확인해야 했다.

2026-10-23 Kaggle T4 두 장(jeju-k3)에서 이 문서 끝의 스크립트로 쟀다. 문장 단위 test에서 512건을 뽑아(seed 0)
같은 병합본을 fp16 연산과 fp32 연산으로 돌렸다. 원 문장은 출력하지 않았다.

## 1. fp16 평가값은 유효하다

| 항목 | fp16 | fp32 |
|---|---:|---:|
| MLP 출력에 inf·nan이 생긴 프롬프트 (36층 합계) | 0 | 0 |
| 마지막 위치 logits에 inf·nan이 생긴 프롬프트 | 0 | 0 |
| 정확도 (GPT-4o 라벨, 512건) | 0.5137 | 0.5137 |
| 두 연산의 라벨 일치 | 511 / 512 | |

MLP 출력 최대 절댓값이 100을 넘은 층은 넷이다. 나머지 32개 층은 80 아래였다.

| 층 | fp16 | fp32 |
|---:|---:|---:|
| 6 | 7,128.0 | 7,127.4 |
| 16 | 220.1 | 219.9 |
| 34 | 1,444.0 | 1,446.1 |
| 35 | 4,416.0 | 4,427.0 |

가장 큰 값(7,128)도 fp16 최댓값 65,504의 11%다. fp16 추론에서는 넘침이 없었고, 라벨도 거의 같으므로
fp16 평가값 0.4738을 그대로 쓴다. 이 진단은 transformers로 했고 평가는 vLLM으로 했으므로 연산 커널은 다르다.
같은 가중치를 fp16으로 돌려도 활성값이 범위 안에 있다는 근거로 쓴다.

## 2. 양자화 실패는 오차 계산에서 났다

llm-compressor 0.14.0의 AWQ는 격자 탐색에서 원 출력과 양자화 출력의 차이를
`torch.nn.functional.mse_loss(fp16_batch, int_w_batch, reduction="sum")`으로 합한 뒤 원소 수로 나눈다
(`modifiers/transform/awq/base.py`, `_compute_loss`). 입력이 fp16이면 합도 fp16 텐서로 나온다.
로컬 torch 2.14.1에서 64×2560 원소에 차이 4씩을 주면 합이 `inf`가 됐다(제곱합 약 262만).

출력 크기가 수천인 35번 층에서는 양자화 오차의 제곱합이 65,504를 넘기 쉽다. 보정을 fp32로 하자
(`--dtype float32`, fe33910) 같은 설정으로 끝났다. 층별 오차 값을 직접 찍어 보지는 않았으므로,
위 설명은 실패 로그와 라이브러리 코드로 본 원인이다. 6번 층은 출력이 더 컸지만 fp16 보정을 통과했다.

## 3. fp32 보정 AWQ 결과

| 구성 | macro-F1 | 정확도 | 독립 재라벨 F1 | 처리량 (건/s) | p95 |
|---|---:|---:|---:|---:|---:|
| fp16 | 0.4738 | 0.5113 | 0.4002 | 50.6 | 1,345ms |
| AWQ (fp32 보정) | 0.4618 | 0.4967 | 0.3729 | 52.1 | 1,293ms |

AWQ − fp16 macro-F1 차이는 −0.0120, 부트스트랩 95% 구간 −0.0198 ~ −0.0043이다
(`compare_lora_qwen3-4b_awq__lora_qwen3-4b_fp16.json`). fp32 보정은 T4 한 장에서 47분 걸렸다.
실패한 fp16 보정은 마지막 층에서 멈출 때까지 14분이었다.

## 스크립트

Kaggle 커널에서 `REPO=/tmp/repo python diag_qwen3_fp16.py 512 /tmp/diag.json`으로 돌렸다.
fp32 4B(16GB)는 T4 한 장에 안 들어가 두 장에 나눠 올렸다. 아래는 그 스크립트에 저장소 포맷터(ruff)만 적용한 것이다.

```python
# Qwen3-4B LoRA 병합본(가중치 fp16)을 fp16 연산과 fp32 연산으로 같은 test 프롬프트에 돌려 비교한다.
#   1) 층마다 MLP 출력에 inf/nan이 생긴 프롬프트 수와 위치(프롬프트 안 몇 번째 토큰인지)
#   2) 층마다 MLP 출력의 최대 절댓값 (fp32 값이 65504를 넘으면 fp16에서 넘친다)
#   3) vLLM choice 제약과 같은 방식(라벨 토큰 트라이 위 탐욕 선택)으로 고른 라벨이 두 연산에서 같은지
# 문장 원문은 출력하지 않는다.
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.environ.get("REPO", "/tmp/repo")
DEV = os.environ.get("DIAG_DEVICE", "cuda:0")
sys.path.insert(0, f"{REPO}/src")
from jeju_lora.data import read_jsonl  # noqa: E402
from jeju_lora.labels import LABEL_NAMES  # noqa: E402
from jeju_lora.prompts import build_messages  # noqa: E402

M = os.environ.get("DIAG_MODEL", f"{REPO}/outputs/merged/qwen3-4b")
N = int(sys.argv[1])
OUT = sys.argv[2]
FP16_MAX = 65504.0

tok = AutoTokenizer.from_pretrained(M)
items = random.Random(0).sample(read_jsonl(f"{REPO}/data/splits_sentence/test.jsonl"), N)
label_ids = [tok(n, add_special_tokens=False)["input_ids"] for n in LABEL_NAMES]
print("label token lengths", [len(x) for x in label_ids], flush=True)


def choose(logits, plen):
    """라벨 토큰 트라이 위에서 매 단계 허용된 토큰 중 argmax (vLLM choice 제약과 같은 탐욕 선택)."""
    alive = list(range(len(label_ids)))
    d = 0
    while len(alive) > 1:
        nxt = {}
        for k in alive:
            if d < len(label_ids[k]):
                nxt.setdefault(label_ids[k][d], []).append(k)
        if not nxt:
            break
        row = logits[alive[0], plen + d - 1]  # 접두가 같은 시퀀스끼리는 이 위치 logits가 같다
        best = max(nxt, key=lambda t: row[t].item())
        alive = nxt[best]
        d += 1
    return alive[0]


def run(model, device):
    n_layers = len(model.model.layers)
    bad_prompts = [0] * n_layers
    bad_pos = [dict() for _ in range(n_layers)]
    maxabs = [0.0] * n_layers
    cur = {}

    def hook(i):
        def h(mod, inp, out):
            o = out[:, : cur["plen"]].float()
            fin = torch.isfinite(o)
            if not fin.all():
                bad_prompts[i] += 1
                for p in (~fin).any(-1).any(0).nonzero().flatten().tolist():
                    bad_pos[i][p] = bad_pos[i].get(p, 0) + 1
            if fin.any():
                maxabs[i] = max(maxabs[i], o[fin].abs().max().item())

        return h

    hs = [layer.mlp.register_forward_hook(hook(i)) for i, layer in enumerate(model.model.layers)]
    preds, nonfinite_logits = [], 0
    with torch.no_grad():
        for it in items:
            prompt = tok.apply_chat_template(
                build_messages(it, "jeju"), tokenize=False, add_generation_prompt=True
            )
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            cur["plen"] = len(p)
            seqs = [p + lab for lab in label_ids]
            L = max(len(s) for s in seqs)
            ids = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in seqs], device=device)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=device)
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
            if not torch.isfinite(logits[0, len(p) - 1]).all():
                nonfinite_logits += 1
            preds.append(choose(logits, len(p)))
    for h in hs:
        h.remove()
    return {
        "preds": preds,
        "prompts_with_nonfinite_mlp_out": bad_prompts,
        "nonfinite_positions": [{str(k): v for k, v in sorted(d.items())} for d in bad_pos],
        "mlp_out_max_abs_finite": maxabs,
        "prompts_with_nonfinite_last_logits": nonfinite_logits,
    }


res = {
    "n": N,
    "row_ids": [it["row_id"] for it in items],
    "gold": [int(it["label"]) for it in items],
}
m16 = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
res["fp16"] = run(m16, DEV)
del m16
torch.cuda.empty_cache()
if DEV.startswith("cuda"):  # fp32 4B(16GB)는 T4 한 장에 안 들어가 두 장에 나눈다
    m32 = AutoModelForCausalLM.from_pretrained(
        M, dtype=torch.float32, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"}
    ).eval()
else:
    m32 = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).to(DEV).eval()
res["fp32"] = run(m32, DEV)

a, b, g = res["fp16"]["preds"], res["fp32"]["preds"], res["gold"]
res["agreement_fp16_fp32"] = sum(x == y for x, y in zip(a, b)) / N
res["acc_fp16"] = sum(x == y for x, y in zip(a, g)) / N
res["acc_fp32"] = sum(x == y for x, y in zip(b, g)) / N
res["layers_fp32_max_over_fp16_range"] = [
    i for i, v in enumerate(res["fp32"]["mlp_out_max_abs_finite"]) if v > FP16_MAX
]
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(
    json.dumps(
        {
            k: res[k]
            for k in [
                "n",
                "agreement_fp16_fp32",
                "acc_fp16",
                "acc_fp32",
                "layers_fp32_max_over_fp16_range",
            ]
        }
    )
)
for k in ("fp16", "fp32"):
    r = res[k]
    print(k, "nonfinite last logits:", r["prompts_with_nonfinite_last_logits"])
    for i, c in enumerate(r["prompts_with_nonfinite_mlp_out"]):
        if c:
            print(f"  layer {i}: {c} prompts, positions {r['nonfinite_positions'][i]}")
    print("  max|mlp out| last 4 layers:", [round(v, 1) for v in r["mlp_out_max_abs_finite"][-4:]])
```

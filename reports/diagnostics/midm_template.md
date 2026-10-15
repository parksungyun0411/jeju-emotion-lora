# Midm-2.0-Mini 채팅 템플릿 진단

2026-10-15 Colab T4(fp16)에서 Midm LoRA 학습이 두 번 실패한 원인을 잰 기록이다. 아래 숫자는 이 문서 끝의
스크립트를 VM에서 그대로 돌린 출력이고, 학습 로그는 `outputs/logs/colab/train_midm_fail_*.log`(저장소 밖)에 있다.
원 문장은 출력하지 않았다. 대응은 [docs/colab_t4.md](../../docs/colab_t4.md)의 '문제와 대응' 3·4.

이 진단은 감정 1·2의 이름이 뒤바뀐 코드(904cdb5에서 정정)로 쟀다. 결론(정답 잘림, 답 끝 토큰 불일치)은 감정 이름과
무관하지만, 손실 값과 kanana 비교 수치는 뒤바뀐 이름으로 학습·측정한 값이다.

## 1. 학습 예시 길이 (train 앞 2,000행, 프롬프트+정답 토큰 수)

| 모델 | 프롬프트 최소/중앙/최대 | 프롬프트+정답 최소/중앙/최대 | 256 초과 |
|---|---|---|---:|
| Midm-2.0-Mini (공식 템플릿) | 557 / 565 / 590 | 559 / 567 / 592 | 2,000 |
| Qwen3-4B | 86 / 97 / 136 | 90 / 101 / 140 | 0 |
| kanana-1.5-2.1b | 87 / 97 / 133 | 90 / 100 / 136 | 0 |

공식 템플릿 그대로 `max_length: 256`으로 학습하면 TRL이 정답이 잘린 예시 50,000개를 모두 버리고
`ValueError: num_samples should be a positive integer value, but got num_samples=0`으로 멈췄다.

## 2. 답 끝 토큰

기본 문구만 뺀 템플릿으로 다시 학습하자 손실이 내려가지 않았다(학습 로그, 20스텝마다):

| epoch | loss | mean_token_accuracy | entropy |
|---:|---:|---:|---:|
| 0.0128 | 10.02 | 0.1625 | 3.359 |
| 0.02559 | 4.142 | 0.192 | 4.549 |
| 0.03839 | 3.377 | 0.245 | 4.693 |
| 0.05118 | 3.273 | 0.2731 | 4.581 |
| 0.06398 | 3.309 | 0.2498 | 4.614 |
| 0.07678 | 3.166 | 0.305 | 4.489 |
| 0.08957 | 3.168 | 0.3077 | 4.515 |
| 0.1024 | 3.175 | 0.2977 | 4.463 |
| 0.1152 | 3.151 | 0.3076 | 4.485 |
| 0.128 | 3.174 | 0.3044 | 4.521 |
| 0.1408 | 3.106 | 0.3285 | 4.436 |
| 0.1536 | 3.107 | 0.3142 | 4.445 |
| 0.1663 | 3.123 | 0.3148 | 4.451 |
| 0.1791 | 3.111 | 0.3249 | 4.456 |
| 0.1919 | 3.094 | 0.3349 | 4.404 |

308/1563스텝(24분 19초)에서 중단했다. kanana 50,000행 학습은 같은 epoch 0.1919에서 손실 0.3686, 정확도 0.8780이었다([reports/train/_invalid_swapped_names/kanana-1.5-2.1b.json](../train/_invalid_swapped_names/kanana-1.5-2.1b.json)의 log_history).

기반 모델(fp16)로 train 8문장의 정답 토큰별 손실을 쟀다(마지막 문장 값과 8문장 첫 토큰 평균):

| 템플릿 | 정답 토큰 | 토큰별 손실 | 첫 토큰(라벨) 평균 손실 |
|---|---|---|---:|
| 공식 | 라벨, `<\|eot_id\|>` | 0.92, 32.21 | 1.319 |
| 기본 문구 제거 | 라벨, `<\|eot_id\|>` | 0.78, 31.63 | 0.995 |

은닉 상태 최댓값은 층마다 938 이하였고 inf·NaN은 없었다(fp16 넘침이 아니다).

train 16문장에서 라벨 바로 뒤 다음 토큰 분포:

| 템플릿 | argmax | `<\|eot_id\|>` 평균 확률 | `<\|end_of_text\|>` 평균 확률 |
|---|---|---:|---:|
| 공식 | `<\|end_of_text\|>` 16/16 | 1.37e-14 | 1.000 |
| 기본 문구 제거 | `<\|end_of_text\|>` 16/16 | 2.47e-14 | 1.000 |

모델 설정도 같다: generation_config `eos_token_id: 2`(`<|end_of_text|>`), `<|eot_id|>`는 131301번.
공식 템플릿이 assistant 차례를 `<|eot_id|>`로 닫는 것과 모델이 실제로 내는 토큰이 다르다.
assistant 차례만 eos로 닫도록 고친 템플릿으로 학습하자 첫 로그(20스텝) 손실은 1.04, 정확도 0.6403이었다
(kanana 50,000행 학습의 첫 로그는 0.8471).

## 스크립트

<details><summary>예시 길이</summary>

```python
import json, sys

sys.path.insert(0, "/content/repo/src")
from transformers import AutoTokenizer
from jeju_lora import compat

compat.apply()
from jeju_lora.prompts import to_sft_record

rows = [
    json.loads(l)
    for _, l in zip(range(2000), open("/content/repo/data/splits_sentence/train.jsonl"))
]
for mid, rev in [
    ("K-intelligence/Midm-2.0-Mini-Instruct", "383eb221c52a32278f1985257b264ade8d982e60"),
    ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    ("kakaocorp/kanana-1.5-2.1b-instruct-2505", "7df4bc35ccd610e451809d7106e1c3cf82bfd44c"),
]:
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    lp, lf = [], []
    for r in rows:
        rec = to_sft_record(r)
        p = tok.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True)
        f = tok.apply_chat_template(rec["prompt"] + rec["completion"], tokenize=False)
        lp.append(len(tok(p, add_special_tokens=False)["input_ids"]))
        lf.append(len(tok(f, add_special_tokens=False)["input_ids"]))
    lp.sort()
    lf.sort()
    print(
        mid,
        "prompt min/med/max",
        lp[0],
        lp[len(lp) // 2],
        lp[-1],
        "full min/med/max",
        lf[0],
        lf[len(lf) // 2],
        lf[-1],
        "full>256:",
        sum(x > 256 for x in lf),
    )
```
</details>

<details><summary>정답 토큰별 손실</summary>

```python
"""Midm fp16 진단: 완성 토큰 손실과 층별 활성값 최대치 (원 문장은 출력하지 않는다)."""

import json, sys, torch

sys.path.insert(0, "/content/repo/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from jeju_lora.prompts import to_sft_record
from pathlib import Path

MID, REV = "K-intelligence/Midm-2.0-Mini-Instruct", "383eb221c52a32278f1985257b264ade8d982e60"
rows = [
    json.loads(l) for _, l in zip(range(8), open("/content/repo/data/splits_sentence/train.jsonl"))
]
tok = AutoTokenizer.from_pretrained(MID, revision=REV)
custom = Path("/content/repo/configs/chat_templates/midm-2.0-mini.jinja").read_text()
model = AutoModelForCausalLM.from_pretrained(MID, revision=REV, dtype=torch.float16).cuda().eval()
print(
    "eos",
    tok.eos_token,
    tok.eos_token_id,
    "pad",
    tok.pad_token,
    tok.pad_token_id,
    "bos",
    tok.bos_token,
)
for name, tpl in [("official", None), ("custom", custom)]:
    losses = []
    for r in rows:
        rec = to_sft_record(r)
        kw = {"chat_template": tpl} if tpl else {}
        p = tok.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True, **kw)
        f = tok.apply_chat_template(rec["prompt"] + rec["completion"], tokenize=False, **kw)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        fi = tok(f, add_special_tokens=False)["input_ids"]
        assert fi[: len(pi)] == pi, "prefix mismatch"
        ids = torch.tensor([fi]).cuda()
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        logits = out.logits[0, len(pi) - 1 : -1].float()
        tgt = ids[0, len(pi) :]
        loss = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        losses.append(loss.tolist())
        hs = [h.abs().max().item() for h in out.hidden_states]
        nonfinite = sum(int(not torch.isfinite(h).all()) for h in out.hidden_states)
    print(
        name,
        "completion tokens",
        [tok.convert_ids_to_tokens(t) for t in tgt.tolist()],
        "loss per token (last ex)",
        [round(x, 2) for x in losses[-1]],
        "mean first-token loss",
        round(sum(l[0] for l in losses) / len(losses), 3),
    )
    print(
        name,
        "hidden max per layer (last ex)",
        [round(x) for x in hs[::6]],
        "nonfinite layers",
        nonfinite,
        "logits finite",
        bool(torch.isfinite(out.logits).all()),
    )
```
</details>

<details><summary>라벨 뒤 다음 토큰</summary>

```python
"""Midm: 라벨 뒤에 모델이 실제로 내는 토큰 (원 문장은 출력하지 않는다)."""

import json, sys, torch

sys.path.insert(0, "/content/repo/src")
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from jeju_lora.prompts import build_messages
from jeju_lora.labels import EMOTION_LABELS

MID, REV = "K-intelligence/Midm-2.0-Mini-Instruct", "383eb221c52a32278f1985257b264ade8d982e60"
rows = [
    json.loads(l) for _, l in zip(range(16), open("/content/repo/data/splits_sentence/train.jsonl"))
]
tok = AutoTokenizer.from_pretrained(MID, revision=REV)
custom = Path("/content/repo/configs/chat_templates/midm-2.0-mini.jinja").read_text()
model = AutoModelForCausalLM.from_pretrained(MID, revision=REV, dtype=torch.float16).cuda().eval()
from collections import Counter

for name, kw in [("official", {}), ("custom", {"chat_template": custom})]:
    top = Counter()
    p_eot = []
    p_eos = []
    for r in rows:
        p = tok.apply_chat_template(
            build_messages(r), tokenize=False, add_generation_prompt=True, **kw
        )
        ids = tok(
            p + EMOTION_LABELS[int(r["label"])], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].cuda()
        with torch.no_grad():
            pr = model(ids).logits[0, -1].float().softmax(-1)
        top[tok.convert_ids_to_tokens(int(pr.argmax()))] += 1
        p_eot.append(pr[131301].item())
        p_eos.append(pr[2].item())
    print(
        name,
        "argmax after label:",
        dict(top),
        "mean P(<|eot_id|>)",
        f"{sum(p_eot) / len(p_eot):.2e}",
        "mean P(<|end_of_text|>)",
        f"{sum(p_eos) / len(p_eos):.3f}",
    )
```
</details>

# Colab T4 실행 기록

2026-10-15에 무료 Google Colab T4 한 장으로 GPU 단계를 실제로 돌린 절차와 결과다.
[runpod_setup.md](runpod_setup.md)는 24GB GPU·bf16 기준 계획이고, 이 문서는 실제로 실행한 것이다.
모든 수치는 `reports/` 아래 파일에서 나온다. 결과 표는 [reports/RESULTS.md](../reports/RESULTS.md).

첫 실행은 감정 1·2의 이름이 뒤바뀐 코드(1=슬픔, 2=행복)로 돌았다. 904cdb5에서 1=행복, 2=슬픔으로 고쳤고
그 결과는 `reports/*/_invalid_swapped_names/`로 옮겼다. 같은 VM에서 kanana 제로샷·퓨샷을 다시 잰 뒤 VM이 끊겼고,
나머지 단계는 다음 날 Kaggle T4에서 돌렸다([kaggle_t4.md](kaggle_t4.md)). 아래 '문제와 대응'의 손실 수치는
첫 실행에서 잰 값이다(원인 진단은 감정 이름과 무관하다).

## 환경

| 항목 | 값 |
|---|---|
| 세션 | `colab new -s jeju-t4 --gpu T4` (Colab CLI 0.7.4, 무료 등급) |
| GPU | Tesla T4 15GB (compute capability 7.5, bf16 없음), 드라이버 580.82.07, CUDA 13.0 |
| VM | Python 3.13.15, CPU 2개, RAM 12GiB, 디스크 여유 71GB |
| 과금 소비율 | `colab usage` → `Usage rate: 1.07/hr` (T4 세션 1개 활성) |

환경은 셋으로 나눴다. 버전은 VM에서 `importlib.metadata.version`으로 확인한 값이다.

| 환경 | 용도 | 주요 버전 |
|---|---|---|
| Colab 기본 파이썬 + trl | 학습·병합·인코더 처리량 | torch 2.11.0+cu130, transformers 5.18.0, peft 0.21.1, trl 1.14.1, accelerate 1.15.0, datasets 4.8.5 |
| `/content/venv-serve` | vLLM 서빙 | vllm 0.30.0, torch 2.13.0, transformers 5.19.0, compressed-tensors 0.17.0, xgrammar 0.2.8 |
| `/content/venv-quant` | AWQ 양자화 | llmcompressor 0.14.0, compressed-tensors 0.19.0, torch 2.14.0, transformers 5.17.0, datasets 5.0.1 |

`requirements/train.txt`의 고정 버전(torch 2.14.0, transformers 5.17.0, peft 0.21.2)은 학습에 쓰지 않았다.
Colab에 이미 있는 torch·transformers·peft를 그대로 쓰고 trl만 더했다(설치 시간을 줄이려고).

## 절차

로컬(Mac)에서 `colab` CLI로 VM을 만들고, 코드와 문장 단위 분할만 올렸다. 데이터(AI Hub 기반)는 저장소에 없고
VM에만 올렸다가 세션 종료와 함께 지워진다. 단계마다 결과를 `colab download`로 내려받았다.

```bash
# ── 로컬 ──
colab new -s jeju-t4 --gpu T4
tar czf data_sentence.tgz data/splits_sentence            # train·val·test jsonl
colab upload -s jeju-t4 data_sentence.tgz /content/data_sentence.tgz
colab upload -s jeju-t4 repo.tgz /content/repo.tgz         # git ls-files 기준 코드만

# ── VM (/content/repo) ──
bash scripts/colab_t4.sh setup     # trl, torchao 제거, venv-serve(vllm), venv-quant(llmcompressor), 플러그인 확인
for m in kanana-1.5-2.1b midm-2.0-mini qwen3-4b; do bash scripts/colab_t4.sh baseline $m; done
bash scripts/colab_t4.sh train kanana-1.5-2.1b
bash scripts/colab_t4.sh merge kanana-1.5-2.1b
bash scripts/colab_t4.sh eval kanana-1.5-2.1b fp16
bash scripts/colab_t4.sh train midm-2.0-mini
# … (VM이 끊긴 뒤의 실행 순서와 소요 시간은 kaggle_t4.md)

# ── 로컬 ──
colab stop -s jeju-t4
```

`colab exec`는 한 번에 오래 붙잡고 있으면 `RuntimeError: Connection was lost.`로 끊기는 일이 여러 번 있었다.
그래서 긴 작업은 VM에서 `setsid nohup bash … > /content/logs/<이름>.log &`로 띄우고, `colab exec`는
로그 꼬리를 읽는 짧은 호출로만 썼다. 끊겨도 VM의 작업은 계속 돈다.

## 설계와 달라진 점

| 설계 | 실제 (T4) | 이유·근거 |
|---|---|---|
| bf16 학습·병합·서빙 | fp16 | T4(cc 7.5)에 bf16이 없다. 학습은 기반 가중치 fp16 + AMP fp16(어댑터는 PEFT가 fp32로 둔다), 병합 `--dtype float16`, vLLM `--dtype float16`(로그: `Casting torch.bfloat16 to torch.float16`) |
| 학습 데이터 100%(train 84,307행) 1 epoch | 50,000행(seed 42 무작위) 1 epoch | kanana 전체 1 epoch가 2,635스텝 × 3.92~4.39 s/스텝, 20스텝 시점 tqdm 예상 2:50:47. 2.5h 상한을 넘어 줄였다. 세 모델 모두 같은 50,000행 |
| 원 행 단위 분할 | 문장 단위 분할(test 10,496건) | 원 분할은 test의 70.4%가 train과 같은 문장이다([reports/leakage.md](../reports/leakage.md)) |
| Midm 공식 채팅 템플릿 | 기본 시스템 문구를 빼고 assistant 차례를 eos로 닫은 템플릿([configs/chat_templates/midm-2.0-mini.jinja](../configs/chat_templates/midm-2.0-mini.jinja)) | 아래 '문제와 대응' 3·4. LoRA 학습·병합본·AWQ 보정에만 쓴다. 제로샷·퓨샷은 공식 템플릿 그대로다 |
| requirements/train.txt 고정 버전 | Colab 기본 torch 2.11.0+cu130·transformers 5.18.0·peft 0.21.1 + trl 1.14.1 | 설치 시간을 줄이려고. 양자화만 llmcompressor 0.14.0이 끌어온 torch 2.14.0·transformers 5.17.0 |
| 인코더 처리량 bf16 | fp16 | `bench_encoder`가 GPU의 compute capability로 고른다. 처음에는 `is_bf16_supported()`로 골라 T4에서 bf16 에뮬레이션이 됐다(52f7615에서 고침, [kaggle_t4.md](kaggle_t4.md) 문제 3) |

## 문제와 대응

1. **`colab exec` 연결 끊김.** 수 분 넘게 붙잡는 호출은 `RuntimeError: Connection was lost.`로 자주 끝났다.
   VM 작업은 영향이 없었다. 긴 작업은 VM에서 `setsid nohup`으로 띄우고 로그만 짧게 읽었다.
2. **peft와 Colab 기본 torchao.** 첫 학습이 LoRA 주입 단계에서 멈췄다.
   `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`
   (peft/tuners/lora/torchao.py `dispatch_torchao`). 이 저장소는 torchao를 쓰지 않아 지웠다(`setup` 단계).
3. **Midm 정답 잘림.** Midm 학습이 `ValueError: num_samples should be a positive integer value, but got num_samples=0`으로 멈췄다.
   Midm 공식 템플릿은 시스템 메시지를 줘도 기본 문구(날짜, Mi:dm 소개·지침)를 항상 앞에 붙인다.
   train 앞 2,000행을 각 토크나이저로 재면 학습 예시(프롬프트+정답) 길이는 다음과 같았다.

   | 모델 | 최소 | 중앙값 | 최대 | 256 초과 |
   |---|---:|---:|---:|---:|
   | Midm-2.0-Mini (공식 템플릿) | 559 | 567 | 592 | 2,000 |
   | Qwen3-4B | 90 | 101 | 140 | 0 |
   | kanana-1.5-2.1b | 90 | 100 | 136 | 0 |

   `max_length: 256`에서 모든 예시의 정답이 잘렸고 TRL이 50,000개를 전부 버렸다. `max_length`를 600 이상으로
   올리는 방법은 예시당 토큰이 중앙값 기준 5.7배(567 대 kanana 100)가 되어 시도하지 않았고,
   기본 문구만 뺀 템플릿을 썼다.
   TRL이 예시를 하나라도 버리면 학습 전에 멈추도록 `train_lora`에 검사를 넣었다.
4. **Midm 답 끝 토큰.** 3을 고친 뒤 학습 손실이 10.02에서 시작해 260스텝까지 3.1 근처(정확도 0.31)에 머물렀다.
   fp16으로 한 예시씩 재 보니 라벨 토큰 손실은 0.78~0.92인데 뒤따르는 `<|eot_id|>`의 손실이 32 안팎이었다.
   공식 템플릿은 assistant 차례를 `<|eot_id|>`로 닫지만, 모델은 답 뒤에 `<|end_of_text|>`를 낸다
   (generation_config `eos_token_id: 2`, train 16문장에서 라벨 뒤 argmax 16/16, 평균 확률 1.000, `<|eot_id|>`는 1e-14 수준).
   assistant 차례만 eos로 닫도록 템플릿을 고친 뒤 첫 로그(20스텝) 손실은 1.04로, kanana 50,000행 학습의 첫 로그(0.8471)와
   비슷하게 시작했다. 측정값과 스크립트는 [reports/diagnostics/midm_template.md](../reports/diagnostics/midm_template.md).
   평가에서는 vLLM의 choice 제약이 답 뒤에서 생성을 끝내므로 끝 토큰은 결과에 영향이 없다.

# Kaggle T4 실행 기록

2026-10-16에 Kaggle 무료 GPU(T4 두 장)로 GPU 단계를 돌린 절차와 결과다.
전날 Colab VM이 끊겨 결과 일부를 잃었고([colab_t4.md](colab_t4.md)), 이날 오전에는 Colab 무료 T4 할당이 503으로 거절돼 Kaggle로 옮겼다.
모든 수치는 `reports/` 아래 파일에서 나온다. 결과 표는 [reports/RESULTS.md](../reports/RESULTS.md).

## 환경

| 항목 | 값 |
|---|---|
| 실행 | Kaggle 스크립트 커널, `kaggle kernels push --accelerator NvidiaTeslaT4` (Kaggle CLI 2.2.4), 비공개 |
| GPU | Tesla T4 15GB × 2 (compute capability 7.5, bf16 없음), 드라이버 580.178.04, CUDA 13.0 |
| VM | Python 3.13.15, CPU 4개, RAM 31GB, `/kaggle/working` 20GB |
| 한도 | 주 30시간. T4 두 장 세션도 1시간에 1시간씩 깎인다. GPU 커널 두 개를 동시에 돌릴 수 있었다 |
| 계정 조건 | 휴대폰 인증을 해야 GPU와 인터넷이 열린다 |

환경은 넷으로 나눴다. 버전은 설치 로그와 `importlib.metadata`로 확인한 값이다.

| 환경 | 용도 | 주요 버전 |
|---|---|---|
| 기본 파이썬 + 학습 패키지 | 학습·병합·인코더 처리량 | torch 2.11.0+cu128, transformers 5.18.0, peft 0.21.1, trl 1.14.1, accelerate 1.15.0, datasets 4.8.5 (torchao 제거) |
| `/tmp/vo` (`--system-site-packages`) | 원 인코더 코드 | 기본 torch 2.11.0+cu128, transformers 4.57.6, tokenizers 0.22.2 (로컬 `.venv-orig`와 같은 버전) |
| `/tmp/vs` | vLLM 서빙 | vllm 0.30.0, torch 2.13.0, transformers 5.20.0 |
| `/tmp/vq` | AWQ 양자화 | llmcompressor 0.14.0, torch 2.14.0, transformers 5.17.0 |

## 절차

데이터(AI Hub 기반)는 저장소에 없다. 원 라벨링 파일, 문장 단위 분할(`data_sentence.tgz`), 로컬에서 학습한 Dual KR-BERT의 test 확률,
원 코드 `jeju_kr_bert.py`를 **비공개** Kaggle 데이터셋으로 올려 커널에 붙였다. 측정이 끝나면 데이터셋을 지운다.
커널은 GitHub에서 고정한 커밋을 받아 돌리고, 단계마다 결과를 `/kaggle/working/out`에 복사한다.

```bash
# ── 로컬 ──
kaggle datasets create -p dataset/          # 비공개
kaggle kernels push -p k1/ --accelerator NvidiaTeslaT4
kaggle kernels status <id>                   # RUNNING → COMPLETE
kaggle kernels output <id> -p out/k1         # 끝난 뒤에만 받을 수 있다
```

한 커널 안에서 GPU 두 장을 나눠 썼다. `CUDA_VISIBLE_DEVICES`로 작업마다 한 장씩 주고, vLLM 서버를 동시에 띄울 때는
`PORT`로 포트를 나눴다(`scripts/colab_t4.sh`).

## 실행 순서와 소요 시간

### jeju-k1 (커밋 d58c58e, 02:42 UTC 시작, 4시간 57분)

| GPU | 단계 | 시작 (UTC) | 소요 |
|---|---|---|---:|
| 1 | 학습 패키지 설치 (기본 파이썬) | 02:43 | 38초 |
| 1 | vLLM 환경 | 02:44 | 53초 |
| 0 | E1: 문장 단위 KoELECTRA 7 epoch (fp32) + 앙상블 | 02:44 | 1시간 10분 |
| 1 | kanana LoRA 학습 (50,000행, 1,563스텝) | 02:44 | 1시간 57분 |
| 1 | 병합 → fp16 평가·부하 측정 | 04:42 | 7분 |
| 1 | 인코더 처리량 (bf16 에뮬레이션, 아래 문제 3) | 04:49 | 1분 |
| 1 | AWQ 양자화 → 평가·부하 측정 | 04:51 | 15분 |
| 1 | Midm·Qwen3 제로샷·퓨샷 | 05:05 | 25분 |
| 0 | E2: 원 행 단위 Dual KR-BERT 7 epoch (fp32) | 03:54 | 2시간 28분 |
| 0 | E2: 원 행 단위 KoELECTRA 7 epoch (fp32) + 앙상블 | 06:22 | 1시간 17분 |

kanana 학습은 `train_runtime` 6,815초(4.36 s/스텝), 학습 손실 0.272, 검증 손실 0.375였다
([reports/train/kanana-1.5-2.1b.json](../reports/train/kanana-1.5-2.1b.json)).

### jeju-kb1 (커밋 52f7615, 07:50 UTC)

인코더 처리량만 fp16으로 다시 쟀다. GPU 한 장, 다른 작업 없이 돌았다.

### jeju-k2 (커밋 02af434, 07:54 UTC 시작)

Midm LoRA용 템플릿 기준선과 Midm·Qwen3-4B LoRA(학습 → 병합 → fp16 평가 → AWQ → 평가)를 두 GPU에서 동시에 돌린다.
결과가 나오면 이 절을 채운다.

## 설계와 달라진 점

| 설계 | 실제 (Kaggle T4) | 이유·근거 |
|---|---|---|
| 원 코드 `--fp16`(bf16 autocast) | KoELECTRA·행 단위 Dual 모두 fp32 | T4에 bf16이 없고, 원 코드에는 GradScaler가 없어 fp16 autocast로 바꿔 쓰지 않았다 |
| 문장 단위 앙상블을 한 장치에서 | Dual은 로컬 Apple M4(MPS, bf16 autocast), KoELECTRA는 T4(fp32) | Dual은 10-15에 로컬에서 끝났다. 같은 test 확률(npz, md5 8073048a…)을 올려 앙상블했다 |
| 학습 환경을 venv로 분리 | 기본 파이썬에 직접 설치 | 아래 문제 1 |

## 문제와 대응

1. **venv의 torch와 기본 torchvision 불일치.** `--system-site-packages` venv에 transformers를 설치하자 uv가 torch를 새로 깔았고,
   기본 torchvision과 어긋나 `RuntimeError: operator torchvision::nms does not exist`로 transformers import가 깨졌다(점검 커널).
   학습 패키지는 `uv pip install --system --break-system-packages`로 기본 파이썬에 넣었다. 원 인코더용 `/tmp/vo`는
   GPU 작업을 나누기 전에 만들어, 학습 패키지 설치가 원 코드 실행 도중에 끼어들지 않게 했다.
2. **실행 중에는 결과를 받을 수 없다.** `kaggle kernels output`은 커널이 끝난 뒤에만 파일을 준다.
   단계 로그(`steps.log`)를 표준 출력으로도 내보내 커널 페이지에서 진행을 확인했다. 본 실행 전에 10분 안팎의 점검 커널을 먼저 돌렸다.
3. **T4에서 bf16 에뮬레이션.** 인코더 처리량을 처음 잴 때 정밀도가 `bf16`으로 기록됐다(103.9건/s).
   `torch.cuda.is_bf16_supported()`가 T4에서도 에뮬레이션으로 `True`를 돌려줬기 때문이다(kb1 로그: `cap (7, 5) bf16_supported True`).
   compute capability 8.0 이상에서만 bf16을 쓰도록 고친 뒤(52f7615) fp16으로 다시 재니 717.4건/s였다. 첫 값은 버렸다.
4. **두 GPU에서 vLLM 동시 실행.** `scripts/colab_t4.sh`가 포트 8000에 고정돼 있어 서버 두 개를 동시에 띄울 수 없었다.
   `PORT`를 받아 `predict`·`loadtest`에 `--base-url`로 넘기게 했다(02af434).
5. **무해한 경고.** vLLM·양자화 venv에서 `Error in sitecustomize … No module named 'wrapt'`가 찍힌다.
   Kaggle 기본 파이썬의 sitecustomize가 venv에 없는 wrapt를 불러서 나는 메시지이고, 서빙·양자화 결과에는 영향이 없었다.

## 측정 조건

- LLM 처리량은 vLLM OpenAI 호환 서버에 동시 요청 64개로 test 앞 5,000건을 보내 쟀다. 인코더는 배치 128 순전파로 쟀다.
  인코더 쪽에는 HTTP 서버 오버헤드가 없다.
- kanana fp16·AWQ 부하 측정은 다른 GPU에서 E2 학습이 돌던 시간에 했다(CPU 4개 공유). 인코더 처리량(kb1)은 다른 작업 없이 쟀다.
- 원 코드는 앙상블 가중치 α를 test에서 고른다(문장 단위 0.50, 원 행 단위 0.55). 앙상블 값은 그만큼 낙관적이다.
- $/100만 건은 Colab 종량제 요금 $9.99/100 컴퓨팅 단위(2026-10-15 조회)에 T4 세션 소비율 1.07단위/h
  (2026-10-15 `colab usage` 실측)를 곱한 $0.1069/h로 계산했다. Kaggle 실행 자체는 무료다.

```bash
make report GPU_NAME="Tesla T4" GPU_USD_PER_HOUR=0.1069 \
  PRICE_NOTE='Colab 종량제 100 컴퓨팅 단위 9.99달러(2026-10-15 조회) × T4 세션 소비율 1.07단위/h(2026-10-15 colab usage 실측). 측정은 Kaggle 무료 T4에서 했다'
# PRICE_NOTE에 $를 쓰면 make가 변수로 읽는다
```

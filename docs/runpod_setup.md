# GPU 임대 서버 실행 가이드

이 저장소의 GPU 단계는 아직 한 번도 실행하지 않았다. 아래 순서는 코드 기준으로 정리한 계획이며,
실행하면서 막히는 곳은 이 문서에 고쳐 적는다.

## 서버 조건

| 항목 | 값 |
|---|---|
| GPU | VRAM 24GB 1장 (RTX 4090 기준으로 계획) |
| 이미지 | CUDA 12.x 드라이버가 있는 Ubuntu (RunPod PyTorch 템플릿 등) |
| 디스크 | 영구 볼륨 100GB 이상 — 모델 3종 × (원본 + 병합본 + AWQ) |
| 네트워크 | Hugging Face 다운로드 가능 (세 모델 모두 공개, 토큰 불필요) |

## 예상 GPU 시간·비용 (계획값, 측정 아님)

| 단계 | 내용 | 예상 시간 |
|---|---|---:|
| 준비 | 환경 설치, 모델 다운로드 | 0.5h |
| ① | 원 프로젝트 Dual KR-BERT·KoELECTRA 재학습 + 앙상블 | 1h |
| ③ | 제로샷·퓨샷 3모델 × 2 (test 25,463건) | 1h |
| ④ | LoRA 학습 3모델 × 1 epoch (train 89,118건) | 2.5h |
| ⑥ | 병합 + AWQ 양자화 3모델 | 1h |
| ⑥⑦ | vLLM 평가·부하 측정 3모델 × 2(bf16·AWQ), 인코더 처리량 | 1.5h |
| 여유 | 재시도·디버깅 | 2h |
| 합계 | | 약 9.5h |

비용은 `시간 × 시간당 요금`이다. 커리큘럼 계획에서는 RTX 4090 기준 약 $3~7로 잡았다.
대여 직전에 요금표를 확인하고 승인받은 뒤 시작한다. 실제 요금은 `make report GPU_USD_PER_HOUR=…`에 넣는다.

## 데이터

원 프로젝트의 라벨링 결과 `merged_jeju_balanced_sorted.xlsx`(127,324행, 열: 제주어 문장 / 표준어 문장 / 감정번호 / gpt감정)가 필요하다.
원천은 AI Hub 「한국어 방언 발화(제주도)」이고 라벨은 GPT-4o로 붙였다. AI Hub 약관상 재배포하지 않으므로 이 저장소에는 없다.

- 원 저장소 스크립트가 가리키는 위치: `jeju-emotion-analysis/../2학기_감정분류/merged_jeju_balanced_sorted.xlsx`
- 2026-10-03 기준 이 Mac에서는 찾지 못했다(`~/Desktop/Projects/Datasets/제주어`에는 AI Hub 원천 zip만 있다).
- 파일을 찾으면 `data/raw/merged_jeju_balanced_sorted.xlsx`에 두고 GPU 서버로는 `scp`로 옮긴다.
- 파일을 찾지 못하면 같은 분할을 재현할 수 없다. 이 경우 비교 전제가 깨지므로 진행 전에 계획을 다시 정한다.

## 실행 순서

로컬(Mac)에서 할 일과 GPU 서버에서 할 일을 나눴다. 사람 검수는 GPU를 켜지 않은 상태에서 한다.

```bash
# ── 로컬 ──
make setup-dev
make split                      # data/splits/*.jsonl, reports/split_manifest.json
make verify-split               # 원 저장소 분할 함수와 row_id 순서까지 대조 + 원 보고서 크기(89,118/12,732/25,463) 확인

# ── GPU 세션 1 (약 1h) ──
git clone <이 저장소> && cd jeju-emotion-lora
git clone https://github.com/parksungyun0411/jeju-emotion-analysis ../jeju-emotion-analysis
scp data/raw/merged_jeju_balanced_sorted.xlsx <서버>:jeju-emotion-lora/data/raw/
bash scripts/runpod_bootstrap.sh
make split verify-split ORIG=../jeju-emotion-analysis
make original ORIG=../jeju-emotion-analysis         # runs/original/*.npz, step7_ensemble 보고서
make encoder-eval ALPHA=<step7_ensemble 보고서의 α>
make bench-encoder GPU_NAME="RTX 4090"
# runs/original, outputs/predictions/encoder_ensemble.jsonl, reports/ 를 로컬로 내려받고 서버를 끈다

# ── 로컬: 사람 검수 (약 4h, GPU 꺼진 상태) ──
make review-sample              # 감정별 40건 × 7 = 280건, 앙상블과 기계 라벨이 다른 문항을 절반까지
make review-label REVIEWER=박성윤
make review-report              # reviews/gold.jsonl, reviews/agreement.json

# ── GPU 세션 2 (약 7h) ──
for m in kanana-1.5-2.1b qwen3-4b midm-2.0-mini; do
  make zeroshot fewshot MODEL=$m
  make train merge quantize MODEL=$m
  make eval-lora MODEL=$m VARIANT=bf16 GPU_NAME="RTX 4090"
  make eval-lora MODEL=$m VARIANT=awq  GPU_NAME="RTX 4090"
done
make report GPU_USD_PER_HOUR=<실제 요금> GPU_NAME="RTX 4090"
# reports/, outputs/lora/*/train_summary.json 을 내려받고 서버를 끈다

# ── 로컬 ──
make rescore                    # 검수가 GPU 세션 뒤에 끝났다면 사람 라벨 지표를 다시 계산
make report GPU_USD_PER_HOUR=<실제 요금> GPU_NAME="RTX 4090"
```

## 먼저 확인할 것 (첫 모델에서)

1. `make train MODEL=kanana-1.5-2.1b` 시작 직후 로그의 스텝당 시간으로 전체 학습 시간을 다시 계산한다. 예상의 1.5배를 넘으면 `max_train_samples`로 줄이고 그 사실을 README에 적는다.
2. `make eval-lora VARIANT=awq`를 처음 돌릴 때 vLLM이 AWQ 체크포인트를 읽는지 본다. 학습 환경의 llm-compressor 0.14.0은 compressed-tensors 0.19.0으로 저장하고, vLLM 0.30.0은 0.17.0을 쓴다. 읽지 못하면 `.venv-train`에 `llmcompressor==0.12.0`(compressed-tensors 0.17.1, transformers ≤5.10.1)을 설치해 양자화만 다시 한다.
3. 원 프로젝트 코드는 transformers 4.x 기준이라 `.venv-orig`(transformers<5)에서 돌린다.

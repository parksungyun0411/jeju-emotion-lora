# GPU 임대 서버 실행 가이드

24GB GPU·bf16 기준으로 세운 계획이다. 실제 GPU 실행은 2026-10-15 Colab T4와 2026-10-16·10-23 Kaggle T4(fp16)에서 했고,
그 절차·버전·막힌 곳은 [colab_t4.md](colab_t4.md)와 [kaggle_t4.md](kaggle_t4.md)에 있다. 아래에서 그 실행으로 사실이 확인된 곳은 고쳐 적었다.

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
- 2026-10-03에는 이 Mac에서 찾지 못했고, 이후 찾아 `data/raw/merged_jeju_balanced_sorted.xlsx`에 두었다(전처리 후 127,307행).
- 원 행 단위 분할은 test 행의 70.4%가 train과 같은 문장이라([reports/leakage.md](../reports/leakage.md)) 주 평가는
  문장 단위 분할(`data/splits_sentence/`)로 한다. GPU 서버에는 분할 jsonl만 옮기면 된다.

## 실행 순서

로컬(Mac)에서 할 일과 GPU 서버에서 할 일을 나눴다. 검수(재라벨)는 GPU를 켜지 않은 상태에서 한다.

```bash
# ── 로컬 ──
make setup-dev
make split                      # data/splits/*.jsonl, reports/split_manifest.json
make verify-split               # 원 저장소 분할 함수와 row_id 순서까지 대조. 원 보고서 크기(89,118/12,732/25,463)와 다르면 경고
                                # (지금 파일은 127,307행이라 89,114/12,731/25,462로 경고가 난다. STRICT=1이면 실패)

# ── GPU 세션 1 (약 1h) ──
git clone <이 저장소> && cd jeju-emotion-lora
git clone https://github.com/parksungyun0411/jeju-emotion-analysis ../jeju-emotion-analysis
scp data/raw/merged_jeju_balanced_sorted.xlsx <서버>:jeju-emotion-lora/data/raw/
bash scripts/runpod_bootstrap.sh
make split verify-split ORIG=../jeju-emotion-analysis
make original SPLIT=sentence ORIG=../jeju-emotion-analysis   # runs/original_sentence/*.npz, step7_ensemble 보고서
make original SPLIT=row ORIG=../jeju-emotion-analysis        # runs/original_row/*.npz (누수 영향 확인용)
make encoder-eval SPLIT=sentence ALPHA=<step7_ensemble 보고서의 α>
make bench-encoder GPU_NAME="RTX 4090"
# runs/original_*, outputs/predictions/encoder_ensemble.jsonl, reports/ 를 로컬로 내려받고 서버를 끈다

# ── 로컬: 검수 (계획. 실제로는 Claude가 블라인드로 재라벨했다, reviews/README.md) ──
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
make rescore                    # 검수가 GPU 세션 뒤에 끝났다면 재라벨 지표를 다시 계산
make report GPU_USD_PER_HOUR=<실제 요금> GPU_NAME="RTX 4090"
```

## 먼저 확인할 것 (첫 모델에서)

1. `make train MODEL=kanana-1.5-2.1b` 시작 직후 로그의 스텝당 시간으로 전체 학습 시간을 다시 계산한다. 예상의 1.5배를 넘으면 `max_train_samples`로 줄이고 그 사실을 README에 적는다.
2. AWQ 체크포인트 호환: llm-compressor 0.14.0(compressed-tensors 0.19.0)이 저장한 W4A16_ASYM 체크포인트를
   vLLM 0.30.0(compressed-tensors 0.17.0)이 그대로 읽었다(2026-10-15 T4, `Using MarlinLinearKernel for CompressedTensorsWNA16`).
   llmcompressor 0.12.0으로 내릴 필요는 없었다.
3. 원 프로젝트 코드는 transformers 4.x 기준이라 `.venv-orig`(transformers<5)에서 돌린다.
4. Midm-2.0-Mini는 공식 채팅 템플릿의 기본 시스템 문구 때문에 학습 예시가 559~592토큰이라 `max_length: 256`에서
   정답이 모두 잘리고, 템플릿의 답 끝 토큰(`<|eot_id|>`)이 모델이 실제로 내는 `<|end_of_text|>`와 다르다.
   `configs/chat_templates/midm-2.0-mini.jinja`를 쓴다(근거: [reports/diagnostics/midm_template.md](../reports/diagnostics/midm_template.md)).
5. kanana-1.5-2.1b는 transformers 5.17·5.18에서 설정 검증(hidden 1792, head 24)에 걸린다. 학습·병합·양자화는
   `jeju_lora.compat`, vLLM은 `vllm.general_plugins` 진입점으로 우회한다(T4에서 둘 다 확인).

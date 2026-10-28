# 단계별 실행. GPU 단계는 scripts/runpod_bootstrap.sh로 .venv-train·.venv-serve를 만든 뒤 쓴다.
# 예: make train MODEL=qwen3-4b
SHELL := /bin/bash
.DEFAULT_GOAL := help

PY       ?= $(firstword $(wildcard .venv/bin/python .venv-serve/bin/python) python3)
ORIG_PY  ?= .venv-orig/bin/python
TRAIN_PY ?= .venv-train/bin/python
SERVE_PY ?= .venv-serve/bin/python
MODEL    ?= qwen3-4b
VARIANT  ?= bf16
DATA     ?= data/raw/merged_jeju_balanced_sorted.xlsx
ORIG     ?= $(HOME)/Documents/GitHub/personal/jeju-emotion-analysis
SPLIT    ?= sentence
ORIG_OUT ?= runs/original_$(SPLIT)
ALPHA    ?=
REVIEWER ?=
GOLD     ?= reviews/gold.jsonl
GPU_NAME ?=
GPU_USD_PER_HOUR ?=
PRICE_NOTE ?=
CONCURRENCY ?= 64

HF_ID    = $(shell $(PY) -m jeju_lora.config $(MODEL) hf_id)
REVISION = $(shell $(PY) -m jeju_lora.config $(MODEL) revision)
MODEL_DIR_bf16 = outputs/merged/$(MODEL)
MODEL_DIR_awq  = outputs/awq/$(MODEL)
TAG = lora_$(MODEL)_$(VARIANT)
SPLIT_DIR_sentence = data/splits_sentence
SPLIT_DIR_row      = data/splits
ENC_TAG_sentence   = encoder_ensemble
ENC_TAG_row        = encoder_ensemble_row

.PHONY: help setup-dev test lint smoke setup-gpu split verify-split leakage unseen-eval split-sentence original encoder-eval \
        review-sample review-label review-report zeroshot fewshot train merge quantize \
        eval-lora bench-encoder report rescore compare

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

# ── 로컬 ────────────────────────────────────────────────
setup-dev: ## 로컬 개발 환경 (.venv, Python 3.12)
	uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev]"

test: ## pytest (스모크는 torch가 있을 때만 실행)
	$(PY) -m pytest -q

lint: ## ruff
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

smoke: ## 작은 랜덤 모델로 학습→병합→생성 경로 확인 (CPU)
	$(PY) -m pytest -q -m smoke

# ── ① 데이터·기존 수치 재현 ─────────────────────────────
split: ## 원 프로젝트와 같은 절차·시드의 행 단위 7:1:2 분할 → data/splits/*.jsonl, reports/split_manifest.json
	$(PY) -m jeju_lora.data --data $(DATA)

verify-split: ## 같은 파일에 원 저장소의 분할 함수를 돌려 row_id 순서까지 대조 (원 보고서와 크기가 다르면 경고, STRICT=1이면 실패)
	$(PY) -m jeju_lora.verify_split --data $(DATA) --original-repo $(ORIG) $(if $(STRICT),--strict-sizes,)

leakage: ## 원 행 단위 분할의 중복 문장 누수 진단 → reports/leakage.{md,json}
	$(PY) -m jeju_lora.leakage --data $(DATA)

unseen-eval: ## 원 행 단위 분할 인코더 예측을 train에 있는/없는 문장으로 나눠 채점 → reports/diagnostics/row_split_seen_unseen.{md,json}
	$(PY) -m jeju_lora.unseen_eval --run-dir runs/original_row

split-sentence: ## 문장 단위 7:1:2 분할 → data/splits_sentence/*.jsonl, reports/split_manifest_sentence.json
	$(PY) -m jeju_lora.sentence_split --data $(DATA)

original: ## 원 인코더 코드를 원 하이퍼파라미터(7 epoch, 배치 64)로 재학습 (SPLIT=sentence|row, AMP=1이면 bf16 autocast, GPU, 원 저장소는 읽기만)
	$(ORIG_PY) -m jeju_lora.run_original --split $(SPLIT) --steps dual koelectra ensemble \
	  --data $(DATA) --original-repo $(ORIG) --out $(ORIG_OUT) $(if $(AMP),--amp,)

encoder-eval: ## 앙상블 test 예측을 row_id에 붙여 재채점 (SPLIT=sentence|row, ALPHA=원 실행에서 고른 값, GOLD=재라벨)
	@test -n "$(ALPHA)" || { echo "ALPHA=값 필요 ($(ORIG_OUT)/step7_ensemble_report.txt 참고)"; exit 1; }
	$(PY) -m jeju_lora.encoder_eval --test $(SPLIT_DIR_$(SPLIT))/test.jsonl \
	  --dual-npz $(ORIG_OUT)/step3_dual_probs.npz \
	  --single-npz $(ORIG_OUT)/branch1_koelectra_probs.npz --alpha $(ALPHA) \
	  --gold $(GOLD) --tag $(ENC_TAG_$(SPLIT))

# ── ② 검수 평가셋 (독립 재라벨) ──────────────────────────────────
review-sample: ## 감정별 층화 + 모델 불일치 우선 280건 추출
	$(PY) -m jeju_lora.review.sample --n 280 \
	  $(if $(wildcard outputs/predictions/encoder_ensemble.jsonl),--predictions outputs/predictions/encoder_ensemble.jsonl,)

review-label: ## 터미널 라벨링 (REVIEWER=이름)
	@test -n "$(REVIEWER)" || { echo "REVIEWER=이름 필요"; exit 1; }
	$(PY) -m jeju_lora.review.label --reviewer "$(REVIEWER)"

review-report: ## 일치도 보고 + reviews/gold.jsonl 생성
	$(PY) -m jeju_lora.review.report

# ── ③ 제로샷·퓨샷 기준선 ────────────────────────────────
zeroshot: ## 기반 모델 제로샷 (MODEL=키)
	scripts/with_vllm.sh $(HF_ID) $(MODEL) --revision $(REVISION) -- \
	  $(SERVE_PY) -m jeju_lora.predict --model $(MODEL) --tag zeroshot_$(MODEL) --shots 0

fewshot: ## 기반 모델 퓨샷, 클래스당 1개 (MODEL=키)
	scripts/with_vllm.sh $(HF_ID) $(MODEL) --revision $(REVISION) -- \
	  $(SERVE_PY) -m jeju_lora.predict --model $(MODEL) --tag fewshot_$(MODEL) --shots 1

# ── ④ LoRA 학습 → ⑥ 병합·양자화 ─────────────────────────
train: ## LoRA 학습 (MODEL=키)
	$(TRAIN_PY) -m jeju_lora.train_lora --config configs/lora_$(MODEL).yaml

merge: ## 어댑터 병합 → outputs/merged/MODEL
	$(TRAIN_PY) -m jeju_lora.merge --model $(MODEL) --adapter outputs/lora/$(MODEL)/adapter \
	  --out $(MODEL_DIR_bf16)

quantize: ## AWQ W4A16 → outputs/awq/MODEL
	$(TRAIN_PY) -m jeju_lora.quantize --model-dir $(MODEL_DIR_bf16) --out $(MODEL_DIR_awq)

eval-lora: ## vLLM으로 F1(기계 라벨·재라벨) + 부하 측정 (MODEL=키 VARIANT=bf16|awq)
	scripts/with_vllm.sh $(MODEL_DIR_$(VARIANT)) $(TAG) -- bash -c '\
	  $(SERVE_PY) -m jeju_lora.predict --model $(TAG) --tag $(TAG) && \
	  $(SERVE_PY) -m jeju_lora.loadtest --model $(TAG) --tag $(TAG) \
	    --concurrency $(CONCURRENCY) --gpu "$(GPU_NAME)"'

# ── ⑦ 건당 비용 ─────────────────────────────────────────
bench-encoder: ## KoELECTRA + KR-BERT 앙상블 처리량
	$(TRAIN_PY) -m jeju_lora.bench_encoder --gpu "$(GPU_NAME)"

report: ## reports/RESULTS.md 생성 (GPU_USD_PER_HOUR=값 GPU_NAME=이름 PRICE_NOTE=요금 근거)
	$(PY) -m jeju_lora.cost --gpu-name "$(GPU_NAME)" --price-note "$(PRICE_NOTE)" \
	  $(if $(GPU_USD_PER_HOUR),--gpu-usd-per-hour $(GPU_USD_PER_HOUR),)

rescore: ## 재라벨이 끝난 뒤 저장된 예측으로 지표 재계산 (GOLD=gold 파일, GPU 불필요)
	$(PY) -c "from jeju_lora.predict import rescore_main; rescore_main()" --gold $(GOLD)

compare: ## 두 구성의 macro-F1 차이 부트스트랩 95% 구간 (A=태그 B=태그) → reports/diagnostics/compare_A__B.json
	@test -n "$(A)" -a -n "$(B)" || { echo "A=태그 B=태그 필요 (outputs/predictions/{태그}.jsonl)"; exit 1; }
	$(PY) -m jeju_lora.compare $(A) $(B)

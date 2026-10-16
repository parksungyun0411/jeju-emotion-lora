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
ORIG_OUT ?= runs/original
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

.PHONY: help setup-dev test lint smoke setup-gpu split verify-split leakage split-sentence original encoder-eval \
        review-sample review-label review-report zeroshot fewshot train merge quantize \
        eval-lora bench-encoder report rescore

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
split: ## 원 프로젝트와 같은 7:1:2 분할 → data/splits/*.jsonl, reports/split_manifest.json
	$(PY) -m jeju_lora.data --data $(DATA)

verify-split: ## 원 저장소의 분할 함수와 row_id 순서까지 대조
	$(PY) -m jeju_lora.verify_split --data $(DATA) --original-repo $(ORIG)

leakage: ## 원 행 단위 분할의 중복 문장 누수 진단 → reports/leakage.{md,json}
	$(PY) -m jeju_lora.leakage --data $(DATA)

split-sentence: ## 문장 단위 7:1:2 분할 → data/splits_sentence/*.jsonl, reports/split_manifest_sentence.json
	$(PY) -m jeju_lora.sentence_split --data $(DATA)

original: ## 원 저장소 파이프라인으로 Dual KR-BERT·KoELECTRA·앙상블 재학습 (GPU, 원 저장소는 읽기만)
	mkdir -p $(ORIG_OUT)
	cd $(ORIG) && $(abspath $(ORIG_PY)) src/jeju_kr_bert.py --step dual --fp16 --epochs 6 --batch-size 64 \
	  --data $(abspath $(DATA)) --output-dir $(abspath $(ORIG_OUT))
	cd $(ORIG) && $(abspath $(ORIG_PY)) src/jeju_kr_bert.py --step koelectra --fp16 --epochs 6 --batch-size 128 \
	  --data $(abspath $(DATA)) --output-dir $(abspath $(ORIG_OUT))
	cd $(ORIG) && $(abspath $(ORIG_PY)) src/jeju_kr_bert.py --step ensemble \
	  --dual-probs $(abspath $(ORIG_OUT))/step3_dual_probs.npz \
	  --single-probs $(abspath $(ORIG_OUT))/branch1_koelectra_probs.npz --output-dir $(abspath $(ORIG_OUT))

encoder-eval: ## 앙상블 test 예측을 row_id에 붙여 재채점 (ALPHA=원 실행에서 고른 값)
	@test -n "$(ALPHA)" || { echo "ALPHA=값 필요 ($(ORIG_OUT)/step7_ensemble 보고서 참고)"; exit 1; }
	$(PY) -m jeju_lora.encoder_eval --dual-npz $(ORIG_OUT)/step3_dual_probs.npz \
	  --single-npz $(ORIG_OUT)/branch1_koelectra_probs.npz --alpha $(ALPHA)

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

#!/usr/bin/env bash
# Colab T4 VM(/content/repo)에서 단계별로 실행한다. 절차와 실측 기록은 docs/colab_t4.md.
#
#   bash scripts/colab_t4.sh setup
#   bash scripts/colab_t4.sh baseline MODEL          # 제로샷 + 퓨샷 (vLLM 서버 1회 기동)
#   bash scripts/colab_t4.sh train MODEL
#   bash scripts/colab_t4.sh merge MODEL
#   bash scripts/colab_t4.sh quantize MODEL
#   bash scripts/colab_t4.sh eval MODEL fp16|awq     # F1 + 부하 측정
#   bash scripts/colab_t4.sh bench-encoder
#
# 환경: 학습·병합은 Colab 기본 파이썬(torch cu130, transformers, peft)에 trl만 더한다.
# 서빙은 vllm을 /content/venv-serve에, 양자화는 llmcompressor를 /content/venv-quant에 따로 둔다
# (vllm과 llmcompressor가 요구하는 compressed-tensors 버전이 다르다).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
SERVE_VENV=${SERVE_VENV:-/content/venv-serve}
QUANT_VENV=${QUANT_VENV:-/content/venv-quant}
SERVE_PY=$SERVE_VENV/bin/python
QUANT_PY=$QUANT_VENV/bin/python
export VLLM=$SERVE_VENV/bin/vllm
GPU_NAME=${GPU_NAME:-Tesla T4}
CONCURRENCY=${CONCURRENCY:-64}
SPLIT=data/splits_sentence

stage=${1:?단계 필요}
model=${2:-}

model_info() { $PY -m jeju_lora.config "$model" "$1"; }

case "$stage" in
  setup)
    uv pip install --system "trl==${TRL_VERSION:-1.14.1}"
    # Colab 기본 torchao 0.10.0이 있으면 peft 0.21.1이 LoRA 주입 단계에서 ImportError를 낸다
    # (0.16.0 초과만 지원). 이 저장소는 torchao를 쓰지 않으므로 지운다
    uv pip uninstall --system torchao
    uv pip install --system --no-deps -e .
    uv venv --python 3.13 "$SERVE_VENV"
    uv pip install --python "$SERVE_VENV" "vllm==${VLLM_VERSION:-0.30.0}" -e .
    # kanana 설정 검증 우회 플러그인이 vLLM에 등록됐는지 확인 (src/jeju_lora/compat.py)
    $SERVE_PY -c "from importlib.metadata import entry_points; \
print([e.name for e in entry_points(group='vllm.general_plugins')])"
    ;;
  baseline)
    # T4는 bf16이 없어 fp16으로 띄운다. 서버 1회 기동에 제로샷·퓨샷을 이어서 잰다
    scripts/with_vllm.sh "$(model_info hf_id)" "$model" --revision "$(model_info revision)" \
      --dtype float16 -- bash -c "
      $SERVE_PY -m jeju_lora.predict --model $model --tag zeroshot_$model --shots 0 \
        --precision fp16 --concurrency $CONCURRENCY --data $SPLIT/test.jsonl --train $SPLIT/train.jsonl &&
      $SERVE_PY -m jeju_lora.predict --model $model --tag fewshot_$model --shots 1 \
        --precision fp16 --concurrency $CONCURRENCY --data $SPLIT/test.jsonl --train $SPLIT/train.jsonl"
    ;;
  train)
    $PY -m jeju_lora.train_lora --config "configs/t4/lora_$model.yaml" \
      --train $SPLIT/train.jsonl --val $SPLIT/val.jsonl
    ;;
  merge)
    $PY -m jeju_lora.merge --model "$model" --adapter "outputs/lora/$model/adapter" \
      --out "outputs/merged/$model" --dtype float16
    ;;
  quantize)
    $QUANT_PY -m jeju_lora.quantize --model-dir "outputs/merged/$model" --out "outputs/awq/$model" \
      --train $SPLIT/train.jsonl
    ;;
  eval)
    variant=${3:?fp16|awq}
    case "$variant" in
      fp16) dir="outputs/merged/$model"; precision=fp16 ;;
      awq) dir="outputs/awq/$model"; precision=awq-w4a16 ;;
      *) echo "variant는 fp16|awq" >&2; exit 2 ;;
    esac
    tag="lora_${model}_${variant}"
    scripts/with_vllm.sh "$dir" "$tag" --dtype float16 -- bash -c "
      $SERVE_PY -m jeju_lora.predict --model $tag --tag $tag --precision $precision \
        --concurrency $CONCURRENCY --data $SPLIT/test.jsonl &&
      $SERVE_PY -m jeju_lora.loadtest --model $tag --tag $tag --concurrency $CONCURRENCY \
        --gpu '$GPU_NAME' --precision $precision --data $SPLIT/test.jsonl"
    ;;
  bench-encoder)
    $PY -m jeju_lora.bench_encoder --gpu "$GPU_NAME" --data $SPLIT/test.jsonl
    ;;
  *)
    echo "알 수 없는 단계: $stage" >&2
    exit 2
    ;;
esac

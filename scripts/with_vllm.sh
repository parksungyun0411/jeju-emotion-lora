#!/usr/bin/env bash
# vLLM 서버를 띄우고, 준비되면 명령을 실행한 뒤 서버를 내린다.
#   scripts/with_vllm.sh <모델 경로 또는 HF ID> <서빙 이름> [vllm 추가 인자...] -- <명령...>
# 예: scripts/with_vllm.sh outputs/awq/qwen3-4b qwen3-4b-awq -- make loadtest ...
set -euo pipefail

MODEL="$1"; NAME="$2"; shift 2
EXTRA=()
while [[ $# -gt 0 && "$1" != "--" ]]; do EXTRA+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
[[ $# -gt 0 ]] || { echo "실행할 명령이 없다" >&2; exit 2; }

VLLM="${VLLM:-.venv-serve/bin/vllm}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/vllm_${NAME}.log"

"$VLLM" serve "$MODEL" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.90 \
  --seed 42 \
  "${EXTRA[@]+"${EXTRA[@]}"}" >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "vLLM이 종료됐다. $LOG 확인" >&2; exit 1; fi
  sleep 5
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "15분 안에 준비되지 않았다" >&2; exit 1; }

echo "vLLM 준비 완료: $NAME ($MODEL)"
"$@"

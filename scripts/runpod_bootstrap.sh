#!/usr/bin/env bash
# GPU 임대 서버(RunPod 등, Ubuntu + CUDA 드라이버) 초기 설정.
# 학습·양자화용 .venv-train과 서빙용 .venv-serve를 분리한다:
# vllm 0.30.0은 compressed-tensors==0.17.0, llmcompressor 0.14.0은 ==0.19.0을 요구해 한 환경에 못 넣는다.
#
#   bash scripts/runpod_bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi 없음 — GPU 서버에서 실행한다" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv-train
uv pip install --python .venv-train -r requirements/train.txt
uv pip install --python .venv-train --no-deps -e .

uv venv --python 3.12 .venv-serve
uv pip install --python .venv-serve -r requirements/serve.txt
uv pip install --python .venv-serve --no-deps -e .

# 원 저장소 재현 코드용: 재현 코드(jeju_kr_bert.py)는 transformers 4.x 기준으로 작성됐다
uv venv --python 3.12 .venv-orig
uv pip install --python .venv-orig "torch==2.14.0" "transformers>=4.40,<5" scikit-learn pandas openpyxl numpy

.venv-train/bin/python -c "import torch; assert torch.cuda.is_available(); print('train torch', torch.__version__, torch.cuda.get_device_name(0))"
.venv-serve/bin/python -c "import vllm; print('vllm', vllm.__version__)"

# 모델 내려받기 용량이 크므로 HF 캐시를 영구 볼륨에 둔다 (RunPod 기본: /workspace)
if [[ -d /workspace ]]; then
  echo 'export HF_HOME=/workspace/hf_cache' >> "$HOME/.bashrc"
  echo "HF_HOME=/workspace/hf_cache 를 ~/.bashrc에 추가했다. 새 셸에서 적용된다."
fi
echo "완료. 다음: docs/runpod_setup.md의 '실행 순서'"

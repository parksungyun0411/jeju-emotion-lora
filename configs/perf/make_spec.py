"""라운드별 커널 spec(json)을 만든다. 모델 revision은 models.json 한 곳에서 읽는다.

python configs/perf/make_spec.py r1 > configs/perf/r1.json
"""

import json
import sys
from pathlib import Path

M = json.loads((Path(__file__).parent / "models.json").read_text())
PIP = ["transformers==5.18.0", "peft==0.21.1", "accelerate==1.15.0", "bitsandbytes==0.50.2"]


def job(name, model, est, *args):
    return {"name": name, "est_min": est, "args": [*M[model], *args]}


def r1():
    large = ["--epochs", "4", "--batch", "32", "--lr", "1.5e-5", "--warmup", "0.1"]
    base = ["--epochs", "5", "--batch", "32", "--lr", "3e-5", "--warmup", "0.1"]
    llm = ["--arch", "llm", "--input", "pair", "--batch", "32", "--lr", "1e-4", "--warmup", "0.05"]
    llm += ["--max-len", "192"]
    gpu0 = [
        job("r1_k2_pair", "k2", 80, *llm, "--epochs", "2", "--evals-per-epoch", "2"),
        job("r1_rl_pair", "rl", 35, "--input", "pair", *large),
        job("r1_rl_std", "rl", 30, "--input", "std", *large),
        job("r1_rl_jeju", "rl", 30, "--input", "jeju", *large),
        job("r1_kc_pair", "kc", 12, "--input", "pair", *base),
        job("r1_ko_dual", "ko", 15, "--arch", "dual", *base),
        job("r1_deb_pair", "deb", 12, "--input", "pair", *base),
    ]
    gpu1 = [
        job(
            "r1_k8_pair",
            "k8",
            150,
            *llm,
            "--qlora",
            "--grad-ckpt",
            "--epochs",
            "1",
            "--evals-per-epoch",
            "3",
            "--eval-batch",
            "32",
        ),
        job("r1_xlmr_pair", "xlmr", 35, "--input", "pair", *large[:4], "--lr", "1e-5"),
        job("r1_ko_pair", "ko", 12, "--input", "pair", *base),
        job("r1_mdeb_pair", "mdeb", 12, "--input", "pair", *base),
    ]
    return {"slug": "jeju-perf-r1", "hours": 4.0, "pip": PIP, "gpus": [gpu0, gpu1]}


if __name__ == "__main__":
    print(json.dumps(globals()[sys.argv[1]](), ensure_ascii=False, indent=1))

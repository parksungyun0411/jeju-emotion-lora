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


def r2():
    """R1 val 근거(장부 R2 행): LLM이 단일 최고(k8 pair 0.5526, 2/3 epoch에서 아직 오르는 중),
    인코더는 std 입력이 최고(rl std 0.537 > pair 0.527 > jeju 0.504),
    앙상블은 k8·kc·rl_std를 골랐다.
    LLM은 길이별 묶음으로 패딩을 줄여 1 epoch를 끝까지 돌리고 입력·시드·모델로 다양성을 낸다.
    k2 OOM(R1, grad-ckpt 없이 batch 32×192)은 grad-ckpt로 고친다. 인코더는 R1과 비교가 깨끗하도록
    길이별 묶음을 쓰지 않고 R-Drop·LLRD·EMA를 한 묶음으로 얹어 시드 3개, R1 설정 다른 시드 1개로
    시드 흔들림을 잰다. 각 GPU 줄 뒤쪽일수록 덜 중요하다(시간이 모자라면 SKIP).
    """
    reg = ["--llrd", "0.9", "--ema", "0.999", "--rdrop", "0.5"]
    large = ["--epochs", "4", "--batch", "32", "--lr", "1.5e-5", "--warmup", "0.1"]
    large_reg = ["--epochs", "4", "--batch", "32", "--lr", "2e-5", "--warmup", "0.1", *reg]
    base = ["--epochs", "5", "--batch", "32", "--lr", "3e-5", "--warmup", "0.1"]
    base_reg = ["--epochs", "5", "--batch", "32", "--lr", "4e-5", "--warmup", "0.1", *reg]
    llm = ["--arch", "llm", "--batch", "32", "--lr", "1e-4", "--warmup", "0.05"]
    llm += ["--max-len", "192", "--group-by-length", "--evals-per-epoch", "2", "--eval-batch", "64"]
    k8 = [*llm, "--qlora", "--grad-ckpt", "--epochs", "1"]
    k2 = [*llm, "--grad-ckpt", "--epochs", "2"]
    gpu0 = [
        job("r2_rl_std_reg", "rl", 45, "--input", "std", *large_reg, "--seed", "42"),
        job("r2_rl_std_reg_s7", "rl", 45, "--input", "std", *large_reg, "--seed", "7"),
        job("r2_rl_std_reg_s13", "rl", 45, "--input", "std", *large_reg, "--seed", "13"),
        job("r2_rl_std_s7", "rl", 25, "--input", "std", *large, "--seed", "7"),
        job("r2_kc_std_reg", "kc", 22, "--input", "std", *base_reg, "--seed", "42"),
        job("r2_xlmr_std", "xlmr", 30, "--input", "std", *large[:4], "--lr", "1e-5"),
        job("r2_k2_pair", "k2", 150, *k2, "--input", "pair", "--seed", "42"),
        job("r2_rl_pair_reg", "rl", 62, "--input", "pair", *large_reg, "--seed", "42"),
        job("r2_k2_std", "k2", 110, *k2, "--input", "std", "--seed", "7"),
    ]
    gpu1 = [
        job("r2_k8_pair", "k8", 300, *k8, "--input", "pair", "--seed", "7"),
        job("r2_k8_std", "k8", 230, *k8, "--input", "std", "--seed", "13"),
        job("r2_mdeb_std", "mdeb", 18, "--input", "std", *base),
        job("r2_deb_std", "deb", 16, "--input", "std", *base[:4], "--lr", "5e-5"),
    ]
    return {"slug": "jeju-perf-r2", "hours": 10.5, "pip": PIP, "gpus": [gpu0, gpu1]}


if __name__ == "__main__":
    print(json.dumps(globals()[sys.argv[1]](), ensure_ascii=False, indent=1))

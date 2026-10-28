"""성능 루프 Kaggle 커널 본문(틀). kernel.py build가 SHA·CODE·SPEC 자리를 채워 run.py로 쓴다.

로컬에서 import하지 않는다. 커널 안에서 코드 압축을 풀고, 데이터(비공개 데이터셋)를 풀고,
패키지를 깐 뒤 GPU마다 작업 목록을 차례로 돌린다. 남은 시간이 예상보다 짧으면 SKIP을 남긴다.
"""

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time

SHA = "__SHA__"
CODE = "__CODE__"
SPEC = json.loads(r"""__SPEC__""")
W = "/kaggle/working"
OUT, LOGS = W + "/out", W + "/logs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
START = time.time()
DEADLINE = START + SPEC["hours"] * 3600
STATUS = {}
PROCS = {}
SHOW = ("EVAL", "DONE", "TIME_LIMIT", "Error", "error:", "step ")
FREEZE = "torch|transformers|peft|accelerate|bitsandbytes|tokenizers|sentencepiece|scikit-learn"


def log(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg
    print(line, flush=True)
    with open(LOGS + "/steps.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, name):
    with open(f"{LOGS}/{name}.log", "w") as f:
        rc = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"{'OK' if rc == 0 else 'FAILED'} {name} rc={rc}")
    return rc


def setup():
    log(f"START sha={SHA} slug={SPEC['slug']} hours={SPEC['hours']}")
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall("/tmp/repo")
    found = subprocess.run(
        "find /kaggle/input -name data_perf.tgz | head -1",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    os.makedirs("/tmp/data", exist_ok=True)
    tarfile.open(found).extractall("/tmp/data")
    log("data " + " ".join(sorted(os.listdir("/tmp/data"))))
    pip = " ".join(f"'{p}'" for p in SPEC.get("pip", []))
    uv = "uv pip install --system --break-system-packages"
    cmd = [
        "set -e",
        "which uv >/dev/null || pip install -q uv",
        f"{uv} {pip}" if pip else "true",
        "(uv pip uninstall --system --break-system-packages torchao || true)",
        f"{uv} --no-deps -e /tmp/repo",
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv",
        f"uv pip freeze --system 2>/dev/null | grep -i -E '^({FREEZE})==' || true",
    ]
    if sh("; ".join(cmd), "setup") != 0:
        sys.exit(1)
    os.environ.update(
        HF_HOME="/tmp/hf",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        USE_TF="0",
        USE_JAX="0",
    )


def worker(g, jobs):
    for job in jobs:
        name = job["name"]
        remaining = (DEADLINE - time.time()) / 60
        est = job.get("est_min", 0)
        if remaining < est * 1.1 + 5:
            log(f"[gpu{g}] SKIP {name} (남은 {remaining:.0f}m < 예상 {est}m)")
            STATUS[name] = {"gpu": g, "status": "SKIP"}
            continue
        log(f"[gpu{g}] START {name} (예상 {est}m, 남은 {remaining:.0f}m)")
        t = time.time()
        cmd = [
            sys.executable,
            "-m",
            job.get("module", "jeju_lora.perf.train_cls"),
            "--name",
            name,
            *job["args"],
            "--data-dir",
            "/tmp/data",
            "--out-dir",
            f"{OUT}/{name}",
            "--time-limit-min",
            str(round(max(5.0, remaining - 15), 1)),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        with open(f"{LOGS}/{name}.log", "w") as f:
            p = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/tmp/repo",
            )
            PROCS[g] = p
            for line in p.stdout:
                f.write(line)
                f.flush()
                if any(k in line for k in SHOW):
                    print(f"[gpu{g}] {line}", end="", flush=True)
            rc = p.wait()
        mins = round((time.time() - t) / 60, 1)
        state = "OK" if rc == 0 else f"FAILED rc={rc}"
        STATUS[name] = {"gpu": g, "status": state, "min": mins}
        log(f"[gpu{g}] {state} {name} {mins}m")


def watchdog():
    while time.time() < DEADLINE + 600:
        time.sleep(30)
    log("HARD_STOP")
    for p in PROCS.values():
        p.kill()


def main():
    setup()
    threading.Thread(target=watchdog, daemon=True).start()
    threads = [threading.Thread(target=worker, args=(g, q)) for g, q in enumerate(SPEC["gpus"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = {
        "sha": SHA,
        "slug": SPEC["slug"],
        "wall_min": round((time.time() - START) / 60, 1),
        "jobs": STATUS,
    }
    with open(W + "/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("ALL_DONE " + json.dumps(summary, ensure_ascii=False))


main()

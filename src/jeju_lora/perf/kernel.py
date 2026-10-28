"""성능 루프 Kaggle 커널·비공개 데이터셋 만들기.

  # 비공개 데이터셋 폴더: train·val은 그대로, test는 라벨을 뺀 사본만 넣는다
  python -m jeju_lora.perf.kernel dataset --out-dir ~/.../perf_kaggle/dataset
  # 커널 폴더: 커밋된 src를 git archive로 묶어 run.py 안에 넣는다(공개 저장소에 올리지 않고 실행)
  python -m jeju_lora.perf.kernel build --spec configs/perf/r1.json --out-dir ~/.../perf_kaggle/r1

spec(json): {"slug": "jeju-perf-r1", "hours": 3.0, "pip": [...], "kernel_sources": [...],
             "gpus": [[job, ...], [job, ...]]}
job: {"name": "...", "est_min": 30, "args": [...], "module": "jeju_lora.perf.train_cls"}
GPU마다 작업을 차례로 돌린다. 남은 시간이 작업 예상 시간보다 짧으면 SKIP을 남긴다.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.perf.inputs import strip_labels

KAGGLE_USER = "seongyunpark0411"
DATASET_SLUG = "jeju-perf-private"
SPLIT_DIR = Path("data/splits_sentence")
DATA_TGZ = "data_perf.tgz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(out_dir: Path, extra: list[Path] | None = None) -> dict:
    """train·val 원본과 라벨을 뺀 test를 묶는다.

    extra는 같이 넣을 비라벨 말뭉치(val·test와 중복 제거를 마친 것).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir / "_stage"
    stage.mkdir(exist_ok=True)
    for name in ("train", "val"):
        (stage / f"{name}.jsonl").write_bytes((SPLIT_DIR / f"{name}.jsonl").read_bytes())
    write_jsonl(strip_labels(read_jsonl(SPLIT_DIR / "test.jsonl")), stage / "test_nolabel.jsonl")
    for p in extra or []:
        (stage / p.name).write_bytes(p.read_bytes())
    tgz = out_dir / DATA_TGZ
    with tarfile.open(tgz, "w:gz") as tf:
        for f in sorted(stage.iterdir()):
            tf.add(f, arcname=f.name)
    meta = {
        "title": DATASET_SLUG,
        "id": f"{KAGGLE_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    manifest = {f.name: sha256(f) for f in sorted(stage.iterdir())}
    manifest[DATA_TGZ] = sha256(tgz)
    for f in stage.iterdir():
        f.unlink()
    stage.rmdir()
    return manifest


def code_archive(repo: Path) -> tuple[str, str]:
    """커밋된 HEAD의 src·pyproject를 tar.gz(base64)로. 작업 트리에 안 올린 변경이 있으면 멈춘다."""
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "src", "pyproject.toml"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"커밋하지 않은 변경이 있다 — 커밋 후 다시:\n{dirty}")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    raw = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD", "src", "pyproject.toml", "README.md"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        with tarfile.open(fileobj=io.BytesIO(raw)) as src:
            for m in src.getmembers():
                tf.addfile(m, src.extractfile(m) if m.isfile() else None)
    return sha, base64.b64encode(buf.getvalue()).decode()


RUN_TEMPLATE = Path(__file__).with_name("kernel_run.py")


def build_kernel(spec_path: Path, out_dir: Path, repo: Path) -> dict:
    spec = json.loads(spec_path.read_text())
    sha, code = code_archive(repo)
    names = [j["name"] for q in spec["gpus"] for j in q]
    if len(names) != len(set(names)):
        raise SystemExit("작업 이름이 겹친다")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = RUN_TEMPLATE.read_text()
    for old, new in (
        ('SHA = "__SHA__"', f'SHA = "{sha}"'),
        ('CODE = "__CODE__"', f'CODE = "{code}"'),
        ('SPEC = json.loads(r"""__SPEC__""")', f'SPEC = json.loads(r"""{json.dumps(spec)}""")'),
    ):
        if run.count(old) != 1:
            raise SystemExit(f"틀에서 {old!r}를 하나만 찾아야 한다")
        run = run.replace(old, new)
    (out_dir / "run.py").write_text(run)
    meta = {
        "id": f"{KAGGLE_USER}/{spec['slug']}",
        "title": spec["slug"],
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{KAGGLE_USER}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": spec.get("kernel_sources", []),
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"sha": sha, "slug": spec["slug"], "jobs": names, "run_py_bytes": len(run)}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--out-dir", required=True)
    d.add_argument("--extra", nargs="*", default=[])
    b = sub.add_parser("build")
    b.add_argument("--spec", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--repo", default=".")
    args = p.parse_args()
    if args.cmd == "dataset":
        out = build_dataset(Path(args.out_dir), [Path(x) for x in args.extra])
    else:
        out = build_kernel(Path(args.spec), Path(args.out_dir), Path(args.repo))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

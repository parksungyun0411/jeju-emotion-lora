"""원 프로젝트의 분할 함수와 이 저장소의 분할이 같은 행을 고르는지 대조한다.

원 저장소의 jeju_kr_bert.py를 import하면 torch·transformers까지 올라오므로
AST에서 stratified_split_712 함수만 꺼내 실행한다. 원 저장소는 읽기만 한다.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from jeju_lora.data import (
    DEFAULT_XLSX,
    SEED,
    ids_digest,
    prepare_frame,
    read_table,
    stratified_split_712,
)

# 원 저장소 위치. 2026-10-14 기준 이 Mac에서는 ~/Documents/GitHub/personal 아래에 있다.
_CANDIDATES = [
    Path.home() / "Documents/GitHub/personal/jeju-emotion-analysis",
    Path.home() / "Desktop/Projects/jeju-emotion-analysis",
]
DEFAULT_ORIGINAL_REPO = next((c for c in _CANDIDATES if c.exists()), _CANDIDATES[0])
ORIGINAL_FILE = "src/jeju_kr_bert.py"
# 원 저장소 docs/bert_improvement_report.md에 적힌 분할 크기(전처리 후 127,313행 기준).
# 지금 라벨링 파일은 전처리 후 127,307행이라 크기가 다르다.
# 재현 코드의 절차·시드로 다시 나눈 분할이다.
ORIGINAL_SIZES = {"train": 89_118, "val": 12_732, "test": 25_463}


def load_original_split_fn(repo: str | Path = DEFAULT_ORIGINAL_REPO):
    src_path = Path(repo) / ORIGINAL_FILE
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "stratified_split_712"
        ),
        None,
    )
    if fn is None:
        raise RuntimeError(f"{src_path}에 stratified_split_712가 없다")
    namespace = {"train_test_split": train_test_split}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src_path), "exec"), namespace)
    return namespace["stratified_split_712"]


def compare(df: pd.DataFrame, original_fn, seed: int = SEED) -> dict:
    """같은 DataFrame에 두 분할 함수를 적용해 split별 row_id 순서까지 비교한다."""
    ours = stratified_split_712(df, seed=seed)
    theirs = original_fn(df, seed=seed)
    result = {}
    for name, a, b in zip(("train", "val", "test"), ours, theirs, strict=True):
        result[name] = {
            "n": len(a),
            "match": a["row_id"].tolist() == b["row_id"].tolist(),
            "sha256": ids_digest(a["row_id"]),
        }
    return result


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_XLSX)
    p.add_argument("--original-repo", default=str(DEFAULT_ORIGINAL_REPO))
    p.add_argument("--manifest", default="reports/split_manifest.json")
    p.add_argument(
        "--strict-sizes", action="store_true", help="분할 크기가 원 보고서와 다르면 실패로 끝낸다"
    )
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} 없음 — 실제 데이터로는 대조할 수 없다.")
    df = prepare_frame(read_table(args.data))
    result = compare(df, load_original_split_fn(args.original_repo))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if Path(args.manifest).exists():
        manifest = json.loads(Path(args.manifest).read_text())
        if manifest["test_row_ids_sha256"] != result["test"]["sha256"]:
            raise SystemExit("split_manifest.json의 test 해시와 다르다")
    sizes = {k: v["n"] for k, v in result.items()}
    if sizes != ORIGINAL_SIZES:
        msg = (
            f"분할 크기가 원 보고서와 다르다: {sizes} != {ORIGINAL_SIZES} "
            f"(원 보고서 {sum(ORIGINAL_SIZES.values()):,}행, 이 파일 {sum(sizes.values()):,}행)"
        )
        if args.strict_sizes:
            raise SystemExit(msg)
        print(f"경고: {msg}")
    if not all(v["match"] for v in result.values()):
        raise SystemExit("원 저장소 분할 함수의 결과와 다르다")
    print("OK: 같은 파일에서 train/val/test row_id 순서까지 원 저장소 분할 함수의 결과와 같다")


if __name__ == "__main__":
    main()

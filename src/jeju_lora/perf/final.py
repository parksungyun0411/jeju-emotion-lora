"""성능 루프 최종 test 평가 (루프를 멈출 때 한 번만 실행).

ensemble.py가 val로만 정한 처방(recipe: 실행 폴더·평가 시점·가중치·클래스 편향)을 각 실행의
test 로짓에 그대로 적용하고, 그때 처음 test 라벨(data/splits_sentence/test.jsonl)을 읽어 채점한다.
결과는 기존 결과표와 같은 형식(outputs/predictions/{tag}.jsonl, reports/metrics/{tag}.json)이라
`make compare A={tag} B=lora_kanana-1.5-2.1b_fp16`으로 현재 최고와 부트스트랩 비교를 낸다.

  python -m jeju_lora.perf.final --recipe reports/perf_loop/final_recipe.json --tag perf_best \
      --i-know-this-is-the-single-test-evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jeju_lora.data import read_jsonl, write_jsonl
from jeju_lora.perf.ensemble import log_softmax
from jeju_lora.predict import DEFAULT_GOLD, DEFAULT_TEST, load_gold, score, split_name


def apply_recipe(recipe: dict) -> tuple[np.ndarray, np.ndarray]:
    """처방을 test 로짓에 적용해 (test_row_id, 예측)을 돌려준다. 라벨은 읽지 않는다."""
    mix, row_ids = None, None
    for m in recipe["members"]:
        z = np.load(Path(m["dir"]) / "logits.npz")
        if row_ids is None:
            row_ids = z["test_row_id"]
        elif not np.array_equal(row_ids, z["test_row_id"]):
            raise ValueError(f"{m['dir']}의 test 순서가 다르다")
        lp = m["weight"] * log_softmax(z["test_logits"][m["eval_index"]])
        mix = lp if mix is None else mix + lp
    return row_ids, (mix + np.asarray(recipe["bias"])).argmax(-1)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--gold", default=DEFAULT_GOLD)
    p.add_argument("--pred-dir", default="outputs/predictions")
    p.add_argument("--metrics-dir", default="reports/metrics")
    p.add_argument("--i-know-this-is-the-single-test-evaluation", action="store_true")
    args = p.parse_args()
    if not args.i_know_this_is_the_single_test_evaluation:
        raise SystemExit("루프 종료 때 한 번만 돈다. 확인 플래그가 필요하다")

    recipe = json.loads(Path(args.recipe).read_text())
    row_ids, pred = apply_recipe(recipe)
    test = {int(r["row_id"]): int(r["label"]) for r in read_jsonl(args.test)}
    if set(test) != set(row_ids.tolist()):
        raise ValueError("처방의 test row_id가 test.jsonl과 다르다")
    rows = [
        {"row_id": int(i), "gold": test[int(i)], "pred": int(pr)}
        for i, pr in zip(row_ids, pred, strict=True)
    ]
    write_jsonl(rows, Path(args.pred_dir) / f"{args.tag}.jsonl")
    out = {
        "tag": args.tag,
        "backend": "perf_loop",
        "recipe": recipe,
        "split": split_name(args.test),
        **score(rows, load_gold(args.gold)),
    }
    path = Path(args.metrics_dir) / f"{args.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out["machine_label"], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()

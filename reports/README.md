# reports

이 폴더의 결과 파일은 모두 스크립트가 만든다. LLM 지표와 처리량은 Colab T4 한 장에서 잰다
(절차·버전·문제는 [docs/colab_t4.md](../docs/colab_t4.md)). 평가는 문장 단위 분할 test 10,496건이다.
감정 이름 정정(904cdb5) 뒤의 처리량·학습 기록은 아직 다시 재는 중이라, 지금 `throughput/`·`train/`에는 격리한 옛 결과만 있다.

| 경로 | 만드는 명령 | 내용 |
|---|---|---|
| `split_manifest*.json` | `make split`, `make split-sentence` | 분할 크기, 감정별 test 건수, row_id 해시 (텍스트 없음) |
| `leakage.{md,json}` | `make leakage` | 원 행 단위 분할의 중복 문장 누수 |
| `metrics/{tag}.json` | `scripts/colab_t4.sh baseline·eval`, `python -m jeju_lora.encoder_eval`(`encoder_*`) | macro-F1·정확도·파싱 실패율·혼동행렬 (GPT-4o 라벨 / 독립 재라벨), 분할·정밀도·백엔드 |
| `throughput/{tag}.json` | `scripts/colab_t4.sh eval·bench-encoder` | 건/s, 지연 p50·p95·p99, 동시 요청 수, GPU, 정밀도 |
| `train/{model}.json` | `scripts/colab_t4.sh train`이 쓴 `outputs/lora/{model}/train_summary.json`의 사본 | 학습 설정, 학습 표본 수, 학습 시간, 손실 곡선 |
| `diagnostics/` | 각 문서 안의 스크립트 | 실행 중 막힌 곳을 잰 기록 |
| `RESULTS.md` | `make report` | 위 파일을 모은 표와 $/100만 건 |

`RESULTS.md`는 손으로 고치지 않는다. 파일이 없는 칸은 '측정 전', `configs/report.yaml`에 `not_run` 사유가 적힌 행은
'측정 안 함'으로 나온다. 독립 재라벨 열은 `make rescore GOLD=reviews/relabel_claude/gold.jsonl`로 다시 계산한다. 이 라벨은 사람이 아니라 Claude가 붙였다([reviews/README.md](../reviews/README.md)).

`metrics/`·`throughput/`·`train/` 아래 `_invalid_swapped_names/`는 감정 1·2의 이름이 뒤바뀐 코드(904cdb5에서 정정)로 만든 결과다. 프롬프트·학습 정답의 감정 이름이 틀렸으므로 결과로 쓰지 않는다(손실 곡선은 참고용).

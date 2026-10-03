# reports

**측정 전.** 이 폴더의 결과 파일은 모두 스크립트가 만든다. 아직 GPU 실행을 하지 않아 비어 있다.

| 경로 | 만드는 명령 | 내용 |
|---|---|---|
| `split_manifest.json` | `make split` | 분할 크기, 감정별 test 건수, row_id 해시 (텍스트 없음) |
| `metrics/{tag}.json` | `make zeroshot·fewshot·eval-lora·encoder-eval` | macro-F1·정확도·파싱 실패율·혼동행렬 (기계 라벨 / 사람 검수 라벨) |
| `throughput/{tag}.json` | `make eval-lora·bench-encoder` | 건/s, 지연 p50·p95·p99, 동시 요청 수, GPU |
| `RESULTS.md` | `make report` | 위 파일을 모은 표와 $/100만 건 |

`RESULTS.md`는 손으로 고치지 않는다. 파일이 없는 칸은 '측정 전'으로 나온다.

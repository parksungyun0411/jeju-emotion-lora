# 라벨 v2 진행 기록

다른 컴퓨터에서 이어 하기 위한 문서다. 2026-10-26 기준이며, 단계가 끝날 때마다 갱신한다.

## 왜 하나

원 프로젝트의 0.84는 행 단위 분할의 누수(test 문장 70.4%가 train에도 있음)에서 나왔고, 문장 단위로
다시 나누면 0.52다([reports/leakage.md](../reports/leakage.md)). 원 라벨(GPT-4o)에는 프롬프트에서 온
편향도 있다. `jeju-emotion-analysis/src/automation.py`의 프롬프트가 "부정적인 내용은 슬픔"이라고
지시했고, "미묘한 감정이라도 있으면 감정으로" 붙이게 했다. 앞뒤 문맥 없이 문장만 보게 했고, API 오류는
중립으로 저장했다.

v2는 같은 원천(AI Hub 「한국어 방언 발화(제주도)」)에서 문장을 다시 뽑아 문맥과 함께 다시 라벨링하고,
누수 없는 분할에서 점수를 정직하게 최대한 올린다. 같은 지침으로 두 번 붙인 라벨의 일치도를 상한
참고값으로 함께 보고한다. **누수를 되살리거나 test로 고르는 방법은 쓰지 않는다.**

## 지금 상태

| 단계 | 상태 | 산출물 |
|---|---|---|
| 문장 풀·분할·표본·배치 | 끝 | `data/v2/items`, `data/v2/batches` (git 제외, 체크섬은 [v2_checksums.sha256](v2_checksums.sha256)) |
| 라벨링 지침 | v2.2 | [v2_labeling_guidelines.md](v2_labeling_guidelines.md) |
| 보정 실행(라벨러 2명 × 같은 400문장) | 끝 | 아래 표 |
| DAPT (KR-BERT + 제주어 토큰 107개) | 끝, 이 컴퓨터에만 있음 | `runs/v2/dapt_krbert` (git 제외) |
| **전체 라벨링** | **시작 전** | `data/v2/labels/{train,evalA,evalB,cueA,cueB}` |
| 불일치 판정(evalC·cueC) | 시작 전 | |
| 확정·인코더 학습·Colab LoRA·보고 | 시작 전 | |

## 데이터

`make v2-data AIHUB_ZIP=<[라벨]제주도_학습용데이터_1.zip 경로>`로 만들고 `make v2-verify`로 확인한다.
같은 zip(sha256 `b813e2a0e043b115691acb508c08aeebef3e0bd91d08d8cb9b11c81a43d68cfb`)이면 200개 배치와
말뭉치가 바이트 단위로 같게 나온다(두 번 만들어 대조함). 원천 문장은 AI Hub 라이선스라 저장소에 넣지
않는다.

| 항목 | 값 |
|---|---|
| 분할 단위 | 대화 파일(`sha256(seed:파일명)`), train 80% · val 10% · test 10% |
| 문장 조건 | 방언 형태가 표준어 풀이와 다르고 10~80자, 알아듣지 못한 구간 없음 |
| 문장 풀 | train 701,476 · val 88,671 · test 93,202 (대화 파일 4,000 · 510 · 531) |
| 표본 | train 50,000 (무작위 70% + 감정 단서어 30%) · val 5,000 · test 5,000 (자연 분포) |
| 보강 평가셋 | val_cue 1,446 · test_cue 2,255 (val·test 파일에서 단서어 묶음별 같은 수) |
| 배치 | train 125 · evalA 25 · evalB 25 · cueA 10 · cueB 10 (400문장씩) |
| DAPT 말뭉치 | train 대화 파일의 발화 1,746,686줄. val·test 표본과 겹침 0 |

val·test(자연 분포·보강셋)는 순서를 다르게 섞은 두 묶음(A·B)으로 따로 라벨링한다. A·B가 다른 문장만
세 번째(C)로 다시 붙이고 다수결로 확정한다(`make v2-finalize`).

## 보정 실행 결과

같은 400문장(evalA/001)을 Opus 라벨러 두 명이 서로 보지 않고 붙였다.

| 방식 | 일치율 | κ | 라벨러 간 macro-F1 |
|---|---:|---:|---:|
| v2.0: 400줄을 한 번에 출력 | 81.5% | 0.38 | 0.39 |
| v2.1: 50줄씩 나눠 쓰고 줄마다 `head` 확인 | 93.8% | 0.77 | 0.81 |

v2.0의 낮은 값은 대부분 **라벨이 한 줄씩 밀린 것**이었다. 250번째 줄쯤부터 한 라벨러의 라벨이 옆 문장에
붙었다. 위치를 한 칸씩 어긋나게 맞춰 보면 그 구간이 다시 맞았다. 그래서 출력 형식을 바꿨고,
`collect.check_batch`가 `head`(문장 앞 8글자)를 대조해 밀린 파일을 거부한다. v2.1에서는 두 라벨러가
모두 high·mid로 확신한 문장의 일치율이 99~100%, 둘 다 low인 30문장은 67%였다.

자연 분포에서 중립은 83~87%이고 공포·혐오는 400문장 중 1~4개였다. 그래서 자연 분포 test를 주 지표로,
보강 평가셋을 감정별 추정을 보태는 보조 지표로 쓴다. 보강셋은 단서어로 뽑아 실제보다 쉬울 수 있다.

## 전체 라벨링 실행 방법

Claude Code에서 에이전트에게 배치 하나씩 맡긴다(라벨러 모델: Opus). 배치마다 아래 지시를 주고,
`<PASS>`·`<NNN>`만 바꾼다. 같은 배치를 A·B로 나눠 붙일 때는 서로의 출력을 읽지 못하게 한다.

```text
You are an emotion annotator for Jeju-dialect Korean utterances.
1. Read the annotation guidelines in full: <repo>/docs/v2_labeling_guidelines.md (v2.2). Follow them exactly, including the output section.
2. Input batch: <repo>/data/v2/batches/<PASS>/<NNN>.jsonl (400 JSON lines; fields id, ctx_before, jeju, std, ctx_after, speaker).
3. Work in 8 chunks of 50 lines. For chunk k: Read just those lines (offset/limit), label each line yourself by reading it, then immediately Write <repo>/data/v2/labels/<PASS>/<NNN>.p{k}.jsonl, one line per input line in the same order: {"id": "...", "head": "<first 8 characters of that line's jeju>", "label": <0-6>, "conf": "high"|"mid"|"low"}. Finish a chunk before reading the next.
4. Do NOT use any program, keyword rule, or script to assign labels. Do NOT read other files under <repo>/data/v2/labels/.
5. Return the number of lines written.
```

배치 하나에 에이전트 토큰이 약 15만 개 든다. 전체는 190배치 남짓에 불일치 판정 몇 배치를 더해, 대략
3천만 토큰이다. 실행이 끝나면 `make v2-check PASS=<PASS>`로 확인하고, 실패한 배치만 다시 돌린다.

불일치 판정: evalA·evalB(cueA·cueB)에서 라벨이 다른 문장을 모아 evalC(cueC) 배치로 만들고, 같은 지시로
세 번째 라벨을 붙인다. 배치 생성 스크립트는 아직 없다(남은 일 1).

## 남은 일

1. 불일치 문장 → evalC·cueC 배치를 만드는 명령 추가
2. 전체 라벨링 → `make v2-check` → 불일치 판정 → `make v2-finalize`
3. 인코더: `make v2-train ENCODER=runs/v2/dapt_krbert INPUT=jeju`(및 `dual`), KoELECTRA,
   val로 고른 α 앙상블. 체크포인트·하이퍼파라미터·α는 val로만 정한다
4. LLM LoRA: `data/splits_v2/*.jsonl`(row_id·jeju·std·label, 기존 LoRA 코드와 같은 형식)로 Colab T4에서
   학습(Colab은 psyreo93@gmail.com)
5. 보고: 자연 분포 test macro-F1(부트스트랩 구간), 보강셋, 라벨러 간 상한. `dual`(전사자 표준어 풀이를
   입력으로 씀)은 실제 서비스 조건이 아니므로 따로 적는다

## 다른 컴퓨터에서 이어 하기

```bash
git fetch && git switch relabel-v2
make setup-dev
make v2-data AIHUB_ZIP="<라벨 zip 경로>" && make v2-verify   # 모두 OK여야 한다
make v2-dapt     # DAPT 모델은 저장소에 없으므로 다시 만든다(M4 약 2시간, CUDA면 더 짧다)
```

그다음 "전체 라벨링 실행 방법"부터 이어 한다.

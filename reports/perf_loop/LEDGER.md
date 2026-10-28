# 성능 루프 장부 (문장 단위 분할 val macro-F1 ≥ 0.84 목표)

이 파일만 읽고 다음 세션이 이어서 돌릴 수 있게 쓴다. 사본: iCloud `Personal/개발_측정_이어하기/jeju_perf_loop/LEDGER.md`.

## 현재 상태 (맨 위 유지)

| 항목 | 값 |
|---|---|
| 현재 최고 val 구성 | 아직 없음 (R1 실행 중) |
| 현재 최고 val macro-F1 | — |
| 이번 주 누적 GPU 시간 | 이 루프 0.2 h (+ R1 실행 중, 상한 4.0 h). 주 시작 2026-10-24 00:00 UTC. 공동 장부 배분(10-28): jeju-perf 몫 약 16 h에서 루프 전 다른 작업 약 3.0 h(kj-k4 2.2 h·minwon-k1 0.9 h)를 빼 보수적으로 약 13 h로 잡는다 |
| 비공개 데이터셋 | `seongyunpark0411/jeju-perf-private` (2026-10-28 02:52 UTC 생성, 루프가 끝나면 `yes \| kaggle datasets delete` 로 삭제). data_perf.tgz sha256 d1e7c2a6…26e3: train.jsonl c1c70f47…, val.jsonl 2752cf8f…, test_nolabel.jsonl 9479ba68… (라벨 없음 확인) |
| 브랜치·작업 트리 | `perf-loop` · `/Users/psyreo/Documents/GitHub/personal/jeju-emotion-lora-perf` (push 금지, 메인 세션이 검토) |
| test 평가 | 0회 (루프 종료 때 1회만) |

멈춤 기록 (2026-10-28 05:05 UTC, 소유자 지시로 중단): jeju-perf-r1은 03:13 UTC에 올렸고 05:05 UTC에 아직 실행 중(상한 4.0 h, 늦어도 07:25 UTC쯤 끝).
R1 결과는 아직 받지 않았다. 다음 세션은 아래 '이어하기' 순서부터.

이어하기
1. `kaggle kernels status seongyunpark0411/jeju-perf-r1` → COMPLETE면
   `kaggle kernels output seongyunpark0411/jeju-perf-r1 -p <작업폴더>/out/r1` (로컬 작업폴더: `_archive/jeju-perf-kaggle/`, 데이터 포함이라 git 밖)
2. `out/r1/summary.json`의 벽시계(`wall_min`)를 이 장부 GPU 표와 공동 장부(`kaggle_gpu_ledger.md`)의 R1 줄에 채운다.
3. `python -m jeju_lora.perf.ensemble <out/r1/out/*> --out reports/perf_loop/r1_val.json` (작업 트리에서 `PYTHONPATH=src`) → 실행별 val macro-F1·클래스별·편향 보정(5겹 oof)·앙상블을 R1 행에 적는다. test 로짓은 읽지 않는다.
4. R1 결과로 아래 계획을 고쳐 R2 spec을 `configs/perf/make_spec.py`에 추가 → 커밋 → `python -m jeju_lora.perf.kernel build --spec configs/perf/r2.json --out-dir <작업폴더>/r2` → 공동 장부에 줄을 쓰고 push.
   새 컴퓨터에서는 비공개 데이터셋 폴더(data_perf.tgz)를 다시 만들 필요가 없다(Kaggle에 이미 있음). 로컬 val 채점에는 `data/splits_sentence/val.jsonl`이 아니라 커널 출력 logits.npz의 val_label을 쓴다.

다음 계획 3개

1. (실행 중) R1: GPU0 kanana-1.5-2.1b LoRA 분류 헤드(pair, 2 epoch) → KLUE RoBERTa-large pair·std·jeju → KcELECTRA pair → Dual-Gated KoELECTRA → DeBERTa-v3-ko pair /
   GPU1 kanana-1.5-8b QLoRA(pair, 1 epoch, epoch당 평가 3회) → XLM-R-large pair → KoELECTRA pair → mDeBERTa pair. 시간이 모자라면 뒤쪽은 SKIP으로 남는다
2. R2(R1 결과로 확정): val 최고 계열을 키운다. LLM이 앞서면 kanana-8b 2 epoch 또는 다른 시드·Qwen3-8B·Midm 분류 헤드로 다양성 확보, 인코더 최고 구성의 다중 시드·R-Drop·LLRD·EMA
3. R3: R1·R2 val 로짓 앙상블(전진 선택)과 클래스 편향 보정(5겹 oof로 낙관 정도 기록), 클래스별 오류 분석. DAPT는 JIT 라이선스(NC) 확인 전엔 train 문장만으로

## 원칙 (위반하면 전부 무효)

- 고르는 것(모델·하이퍼파라미터·앙상블 가중치·임계값·보정)은 val로만. 루프 중 test 지표를 계산하지 않는다.
  커널에는 라벨을 뺀 `test_nolabel.jsonl`만 올린다. 커널은 test 로짓만 저장하고 채점할 수 없다.
- test는 val ≥ 0.84로 멈출 때 또는 소유자가 멈추라고 할 때 한 번만 채점하고, `make compare`로 현재 최고와 부트스트랩 95% 구간을 낸다.
- 분할을 바꾸지 않는다. val·test 문장은 학습·DAPT 말뭉치·토크나이저 어휘·하드 마이닝·의사 라벨에 넣지 않는다.
- 비라벨 제주어 말뭉치를 더하면 공백 정규화 뒤 val·test와 중복을 빼고 겹친 수를 여기 적는다.
- 모델은 모델 카드에서 상업 이용 라이선스를 확인하고 revision을 고정한다(`configs/perf/models.json`). EXAONE 금지.
- 원문·파생 텍스트·텍스트 붙은 예측은 git에 올리지 않는다. Kaggle 업로드는 비공개 데이터셋만.
- 수치는 커널 출력(metrics.json·logits.npz)에서만 옮긴다. 재지 않은 값을 쓰지 않는다.

## 입력에 대한 결정

- GPT-4o 라벨은 원 `automation.py`가 '표준어 문장' 열을 넣어 붙였다(`text_to_classify`가 `표준어 문장` 열).
  원 Dual-Gated KR-BERT도 val·test에서 표준어 열을 입력으로 쓰고, 이 저장소 결과표의 Dual KR-BERT 행도 그렇다.
  그래서 표준어 열을 입력으로 쓰는 구성을 허용하고, 행마다 입력(`jeju`·`std`·`pair`=표준어 앞 문장쌍)을 적는다.
- train은 원 파일의 클래스 균형용 복제(최대 56배)를 포함한 84,307행, 고유 문장 36,733개.
  `dedup`은 고유 문장만, `dup`은 원래 복제 행 그대로 학습한다. val·test는 문장당 1행이라 고유 문장 분포를 따른다.
  val 라벨 분포: 중립 1,631 · 행복 910 · 슬픔 1,304 · 분노 650 · 놀람 348 · 공포 187 · 혐오 218.

## 비라벨 말뭉치

- kakaobrain/jejueo(JIT): GitHub 코드는 Apache-2.0이지만 데이터(Kaggle `bryanpark/jit-dataset`)의 라이선스 메타데이터는 **CC-BY-NC-SA-4.0(비상업)**.
  지시문은 Apache-2.0으로 적었으나 실제와 다르다 → 소유자 확인 전에는 쓰지 않는다(차단 사항). DAPT를 하면 train 문장(제주어·표준어)만 쓴다.

## 라이선스 확인 (2026-10-28, HF API·모델 카드)

| 모델 | revision | 라이선스 | 비고 |
|---|---|---|---|
| klue/roberta-large | 28d911204e90 | CC-BY-SA-4.0 | 모델 카드에 메타데이터 없음 → 카드가 가리키는 KLUE GitHub LICENSE(CC-BY-SA-4.0). 상업 이용 가능(저작자 표시·동일 조건). 소유자 확인 권장 |
| FacebookAI/xlm-roberta-large | c23d21b0620b | MIT | |
| beomi/KcELECTRA-base-v2022 | 87285578b468 | MIT | |
| monologg/koelectra-base-v3-discriminator | 68b30cd259f3 | Apache-2.0 | |
| team-lucid/deberta-v3-base-korean | 47a3b00828c5 | Apache-2.0 | |
| microsoft/mdeberta-v3-base | a0484667b223 | MIT | |
| kakaocorp/kanana-1.5-2.1b-instruct-2505 | 7df4bc35ccd6 | Apache-2.0 | |
| kakaocorp/kanana-1.5-8b-instruct-2505 | c963a5f4f649 | Apache-2.0 | |
| snunlp/KR-BERT-char16424 | — | 없음 | 카드·GitHub(snunlp/KR-BERT) 모두 라이선스 없음 → 루프에서 쓰지 않는다. Dual-Gated는 다른 인코더로 구현 |

## GPU 시간 (커널 벽시계, 커널 하나 = GPU 두 장 한 세션)

| 커널 | 시작(UTC) | 벽시계 h | 이번 주 누적 h(루프) |
|---|---|---:|---:|
| jeju-perf-smoke | 2026-10-28 02:53 | 0.2 | 0.2 |
| jeju-perf-r1 | 2026-10-28 03:13 | (05:05 UTC에 실행 중, 상한 4.0. 끝나면 summary.json wall_min으로 채움) | |

## 라운드

| # | 날짜 | 가설 | 변경 | 커널 | GPU h | val macro-F1 (클래스별) | 유지/폐기 | 메모 |
|---|---|---|---|---|---:|---|---|---|
| R0 | 10-28 | 학습기·커널 경로가 T4에서 돈다 | 작은 표본(512~3,200행)으로 10개 경로 점검 | jeju-perf-smoke (ec76255) | 0.2 | 점검용(val 앞 256~512건) — 성능 비교에 쓰지 않음 | — | 실패 4: kanana-2.1b 설정 검증(토크나이저가 우회보다 먼저 불림)·mDeBERTa fp16 체크포인트(GradScaler)·DeBERTa-ko 토크나이저(transformers 5 Unigram 재구성) → 3ce6c37에서 고침. 참고: kanana-8b QLoRA는 train 960행만으로 val 앞 256건 0.42 |

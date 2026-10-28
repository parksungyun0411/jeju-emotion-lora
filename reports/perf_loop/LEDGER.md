# 성능 루프 장부 (문장 단위 분할 val macro-F1 ≥ 0.84 목표)

이 파일만 읽고 다음 세션이 이어서 돌릴 수 있게 쓴다. 사본: iCloud `Personal/개발_측정_이어하기/jeju_perf_loop/LEDGER.md`.

## 현재 상태 (맨 위 유지)

| 항목 | 값 |
|---|---|
| 현재 최고 val 구성 | 아직 없음 (R0 점검 중) |
| 현재 최고 val macro-F1 | — |
| 이번 주 누적 GPU 시간 | 이 루프 0.0 h (주 시작 2026-10-24 00:00 UTC, 루프 전 다른 작업 약 3.0 h 추정 → 남은 한도 약 27 h) |
| 비공개 데이터셋 | `seongyunpark0411/jeju-perf-private` (루프가 끝나면 삭제. 아직 만들기 전) |
| 브랜치·작업 트리 | `perf-loop` · `/Users/psyreo/Documents/GitHub/personal/jeju-emotion-lora-perf` (push 금지, 메인 세션이 검토) |
| test 평가 | 0회 (루프 종료 때 1회만) |

다음 계획 3개

1. R1 인코더 기준선(val): 입력(표준어 앞 문장쌍·제주어만·표준어만), train 중복 제거 여부, 모델(KLUE RoBERTa-large·XLM-R-large·KcELECTRA·KoELECTRA·DeBERTa-v3-ko·mDeBERTa)·Dual-Gated KoELECTRA를 한 커널에서 비교
2. R2 LLM 분류 헤드 LoRA: kanana-1.5-2.1b(fp16 LoRA)·kanana-1.5-8b(QLoRA), 표준어+제주어 입력
3. R3 val 앙상블·클래스 편향 보정(5겹 교차 추정 함께 기록), 이어서 DAPT·다중 시드

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

## 라운드

| # | 날짜 | 가설 | 변경 | 커널 | GPU h | val macro-F1 (클래스별) | 유지/폐기 | 메모 |
|---|---|---|---|---|---:|---|---|---|

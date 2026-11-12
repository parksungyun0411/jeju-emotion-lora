# 성능 루프 장부 (문장 단위 분할 val macro-F1 ≥ 0.84 목표)

이 파일만 읽고 다음 세션이 이어서 돌릴 수 있게 쓴다. 사본: iCloud `Personal/개발_측정_이어하기/jeju_perf_loop/LEDGER.md`.

## 현재 상태 (맨 위 유지)

| 항목 | 값 |
|---|---|
| 현재 최고 val 구성 | (R1 기준) 단일: kanana-1.5-8b QLoRA pair(r1_k8_pair, 0.666 epoch). 앙상블: k8 0.5·KcELECTRA pair 0.25·RoBERTa-large std 0.25 + 클래스 편향 |
| 현재 최고 val macro-F1 | 단일 0.5526(편향 보정 5겹 oof 0.5641) · 앙상블+편향 in-sample 0.5811 / 5겹 oof 0.5632. **목표 0.84와 약 0.28 차이** |
| 이번 주 누적 GPU 시간 | 주 시작 2026-11-07 00:00 UTC(11-14 09:00 KST에 다시 참). jeju-perf 몫 최대 12 h, 10-28 이후 이 계정 커널 없음 → R2 전 0 h. R2 상한 10.5 h(감시 종료 +10분). 지난 주(10-24) 루프 합계 4.2 h(smoke 0.2 + R1 4.0) |
| 비공개 데이터셋 | `seongyunpark0411/jeju-perf-private` (2026-10-28 02:52 UTC 생성, 루프가 끝나면 `yes \| kaggle datasets delete` 로 삭제). data_perf.tgz sha256 d1e7c2a6…26e3: train.jsonl c1c70f47…, val.jsonl 2752cf8f…, test_nolabel.jsonl 9479ba68… (라벨 없음 확인) |
| 브랜치·작업 트리 | `perf-loop` · `/Users/psyreo/Documents/GitHub/personal/jeju-emotion-lora-perf` (perf-loop만 push, main은 건드리지 않음) |
| test 평가 | 0회 (루프 종료 때 1회만) |

상태 (2026-11-12): R1 결과 수거·집계 끝(`r1_val.json`). R2 커널 `jeju-perf-r2`를 올린다(아래 R2 행·GPU 표에 시각과 실제 시간).

이어하기
1. `kaggle kernels status seongyunpark0411/jeju-perf-r2` → COMPLETE면
   `kaggle kernels output seongyunpark0411/jeju-perf-r2 -p <작업폴더>/out/r2` (로컬 작업폴더: `_archive/jeju-perf-kaggle/`, 데이터 포함이라 git 밖)
2. `out/r2/summary.json`의 벽시계(`wall_min`)를 이 장부 GPU 표와 공동 장부(`kaggle_gpu_ledger.md`)의 R2 줄에 채운다.
3. 작업 트리에서 `PYTHONPATH=src python -m jeju_lora.perf.ensemble ../_archive/jeju-perf-kaggle/out/r2/out/* --out reports/perf_loop/r2_val.json`(R2만),
   R1+R2 합친 것은 `... out/r1/out/* out/r2/out/* --out reports/perf_loop/r12_val.json`. 실행 폴더는 상대 경로로 넘긴다(처방 recipe에 로컬 사용자 경로가 남지 않게, final.py는 작업 트리에서 돌린다). test 로짓은 읽지 않는다.
   새 컴퓨터에서는 비공개 데이터셋 폴더(data_perf.tgz)를 다시 만들 필요가 없다(Kaggle에 이미 있음). 로컬 val 채점에는 `data/splits_sentence/val.jsonl`이 아니라 커널 출력 logits.npz의 val_label을 쓴다.

다음 계획 3개

1. (실행) R2: 아래 R2 행. LLM 1 epoch 완주(k8 pair·k8 std·k2 pair·k2 std)와 인코더 정규화(R-Drop·LLRD·EMA) 다중 시드
2. R3(GPU 불필요): R1·R2 val 로짓 앙상블(전진 선택)과 클래스 편향 보정, 5겹 oof로 낙관 정도 기록, 클래스별 혼동 분석(분노·공포·혐오가 낮다)
3. R4 후보(R2 결과로 확정): 이기는 LLM 입력으로 다른 시드·2 epoch, 또는 다른 LLM(Qwen3-8B·Midm-2.0 분류 헤드, 라이선스·revision 확인 뒤). DAPT는 JIT 라이선스(NC) 확인 전엔 train 문장만으로

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
| klue/roberta-large | 28d911204e90 | CC-BY-SA-4.0 | 모델 카드에 메타데이터 없음 → 카드가 가리키는 KLUE GitHub LICENSE(CC-BY-SA-4.0). 상업 이용 가능(저작자 표시·동일 조건). 지금처럼 쓰되 **소유자 확인 대기 중**(2026-11-12) |
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
| jeju-perf-r1 | 2026-10-28 03:13 | 4.0 (wall_min 240.5) | 4.2 |
| (주 2026-11-07 시작) | | | |
| jeju-perf-r2 | (올릴 때 채움) | (상한 10.5, 끝나면 wall_min으로) | |

## 라운드

| # | 날짜 | 가설 | 변경 | 커널 | GPU h | val macro-F1 (클래스별) | 유지/폐기 | 메모 |
|---|---|---|---|---|---:|---|---|---|
| R0 | 10-28 | 학습기·커널 경로가 T4에서 돈다 | 작은 표본(512~3,200행)으로 10개 경로 점검 | jeju-perf-smoke (ec76255) | 0.2 | 점검용(val 앞 256~512건) — 성능 비교에 쓰지 않음 | — | 실패 4: kanana-2.1b 설정 검증(토크나이저가 우회보다 먼저 불림)·mDeBERTa fp16 체크포인트(GradScaler)·DeBERTa-ko 토크나이저(transformers 5 Unigram 재구성) → 3ce6c37에서 고침. 참고: kanana-8b QLoRA는 train 960행만으로 val 앞 256건 0.42 |
| R1 | 10-28 | 인코더·LLM 기준선, 입력(jeju·std·pair) 비교 | 11개 경로(spec `r1`), 4.0 h 상한 | jeju-perf-r1 (3ce6c37) | 4.0 | 단일 최고 k8 pair **0.5526**(편향 oof 0.5641) · 앙상블+편향 in-sample 0.5811 / oof **0.5632**. 앙상블 클래스별: 중립 .604·행복 .726·슬픔 .635·분노 .540·놀람 .561·공포 .522·혐오 .481 | LLM·std 입력 유지, jeju 단독·Dual·DeBERTa pair 폐기 | 아래 'R1 실행별'. k2 OOM, XLM-R·KoELECTRA pair·mDeBERTa SKIP |
| R2 | 11-12 | (1) LLM은 1 epoch를 끝까지 돌리면 오른다 (2) std 입력이 LLM에서도 낫다 (3) 인코더는 R-Drop·LLRD·EMA로 오른다 (4) 시드·입력·모델을 바꾼 멤버가 앙상블 oof를 올린다 | 아래 'R2 구성' | jeju-perf-r2 | (상한 10.5) | (결과 대기) | | |

### R1 실행별 (val 5,248건, 평가 시점은 val 최고, `reports/perf_loop/r1_val.json`)

| 실행 | 입력 | val macro-F1 | 편향 in-sample | 편향 5겹 oof | 최고 시점(epoch) | 분 | 앙상블 가중치 |
|---|---|---:|---:|---:|---|---:|---:|
| r1_k8_pair (kanana-1.5-8b QLoRA r16) | pair | 0.5526 | 0.5723 | 0.5641 | 0.666 (0.333에서 0.5322) | 239.7 | 0.5 |
| r1_rl_std (KLUE RoBERTa-large) | std | 0.5370 | 0.5483 | 0.5393 | 3 / 4 | 22.3 | 0.25 |
| r1_rl_pair | pair | 0.5271 | 0.5399 | 0.5303 | 3 / 4 | 31.5 | 0 |
| r1_kc_pair (KcELECTRA-base) | pair | 0.5129 | 0.5212 | 0.5071 | 3 / 5 | 12.1 | 0.25 |
| r1_ko_dual (Dual-Gated KoELECTRA) | jeju·std | 0.5083 | 0.5192 | 0.5047 | 3 / 5 | 14.4 | 0 |
| r1_rl_jeju | jeju | 0.5037 | 0.5107 | 0.5007 | 3 / 4 | 22.6 | 0 |
| r1_deb_pair (DeBERTa-v3-ko) | pair | 0.4920 | 0.5024 | 0.4933 | 5 / 5 (끝까지 오름) | 19.0 | 0 |
| r1_k2_pair (kanana-1.5-2.1b LoRA) | pair | 실패 | | | | 3.0 | |
| r1_xlmr_pair·r1_ko_pair·r1_mdeb_pair | pair | SKIP(시간 없음) | | | | | |

- 앙상블 편향(로그 확률에 더함, 중립·행복·슬픔·분노·놀람·공포·혐오): 0, 0, 0.1, 0.9, 0.5, 0.6, 1.0 — 소수 클래스 쪽으로 민다.
- 5겹 oof로 재면 앙상블(0.5632)이 k8 단독+편향(0.5641)보다 낫지 않다. in-sample 0.5811은 같은 val로 고르고 잰 낙관값이다.
- k8 pair: epoch당 평가 3회로 잡았으나 1 epoch가 4.6 h(14.5 s/step, batch 32)라 2회(0.333·0.666 epoch)만 하고 4.0 h 한도로 멈췄다. 선형 감쇠가 끝나기 전이고 아직 오르는 중이었다. 평가 1회(val+test 15,744건, eval-batch 32)에 약 25분.
- k2 pair: 2.2분 만에 불러온 뒤 첫 step에서 CUDA OOM(T4 14.56 GiB 중 13.88 GiB 할당). grad-ckpt 없이 batch 32·max-len 192로 LoRA 7개 모듈의 활성값이 모두 남은 탓.
- 혼동(R1 앙상블+편향, val in-sample, 행=GPT-4o 라벨): 가장 많은 것은 슬픔→중립 269·중립→슬픔 183·중립→행복 156·행복→중립 147·중립→분노 137. 오류 대부분이 중립과 감정의 경계다(독립 재라벨 불일치도 대부분 이 경계였다). 재현율은 공포 0.508·혐오 0.550·분노 0.557이 낮고, 정밀도는 혐오 0.427이 가장 낮다.
- 입력: RoBERTa-large에서 std 0.537 > pair 0.527 > jeju 0.504. 라벨이 표준어 열로 붙은 것과 맞는다. 한 시드씩이라 std·pair 차이(0.01)는 시드 흔들림 안일 수 있다(R2에서 잰다).

### R2 구성 (spec `r2`, `configs/perf/r2.json`, 각 GPU 줄 뒤쪽일수록 덜 중요하고 시간이 모자라면 SKIP)

근거와 고른 것
- LLM이 단일 최고이고 아직 오르는 중 → k8을 1 epoch 끝까지(시드 7, pair) 돌리고, 인코더에서 이긴 std 입력으로 k8을 하나 더(시드 13). std는 길이가 짧아 싸다.
- R1의 k8 1 epoch 4.6 h 중 상당 부분이 무작위 배치 패딩이다(train 글자 수 기준 배치 최대 길이 평균: 무작위 176, 길이별 묶음 113 → 1.56배). 학습기에 `--group-by-length`(50배치 묶음 안에서 길이순 정렬, 배치 순서는 섞음, 가장 긴 배치를 첫 step에)를 넣어 LLM에만 쓴다. 인코더는 R1과 비교가 깨끗하도록 쓰지 않는다.
- k2 OOM 고침: `--grad-ckpt`(층 입력만 남김), 가장 긴 배치가 첫 step이라 OOM이면 바로 드러난다. 커널 환경에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- 평가 비용: LLM은 epoch당 평가 2회, eval-batch 64.
- 인코더: 최고인 RoBERTa-large std에 R-Drop 0.5·LLRD 0.9·EMA 0.999를 한 묶음으로(LLRD로 아래층 학습률이 줄어 최상층 lr 1.5e-5→2e-5) 시드 42·7·13. R1 설정 그대로 시드 7을 하나 더 돌려 시드 흔들림을 잰다(R1 시드 42와 비교).
- 앙상블에 뽑힌 KcELECTRA는 std+같은 정규화로, R1에서 SKIP된 XLM-R-large는 std로(다양성). 남는 시간에 RoBERTa-large pair+정규화, k2 std.
- 뺀 것: rl jeju(0.504)·Dual-Gated(가중치 0)·DeBERTa pair(0.492)·KoELECTRA pair. GPU1 꼬리에 mDeBERTa std·DeBERTa-ko std(lr 5e-5, R1에서 끝까지 오름)만 둔다.
- 다른 LLM(Qwen3-8B·Midm)은 이번에 넣지 않았다: 8B QLoRA 1 epoch가 커널 반쪽(GPU 한 장)을 거의 다 쓰고, models.json에 revision·라이선스 확인이 아직 없다.

| GPU | 순서 | 실행 | 모델·입력 | 설정 | 예상 분 |
|---|---|---|---|---|---:|
| 0 | 1 | r2_rl_std_reg | RoBERTa-large std | 4 ep, lr 2e-5, R-Drop 0.5·LLRD 0.9·EMA 0.999, 시드 42 | 45 |
| 0 | 2 | r2_rl_std_reg_s7 | 〃 | 〃 시드 7 | 45 |
| 0 | 3 | r2_rl_std_reg_s13 | 〃 | 〃 시드 13 | 45 |
| 0 | 4 | r2_rl_std_s7 | RoBERTa-large std | R1 설정(4 ep, lr 1.5e-5), 시드 7 | 25 |
| 0 | 5 | r2_kc_std_reg | KcELECTRA std | 5 ep, lr 4e-5, R-Drop·LLRD·EMA | 22 |
| 0 | 6 | r2_xlmr_std | XLM-R-large std | 4 ep, lr 1e-5 | 30 |
| 0 | 7 | r2_k2_pair | kanana-1.5-2.1b LoRA pair | fp16, grad-ckpt, 2 ep, 길이별 묶음, 시드 42 | 150 |
| 0 | 8 | r2_rl_pair_reg | RoBERTa-large pair | 정규화 묶음, 시드 42 | 62 |
| 0 | 9 | r2_k2_std | kanana-1.5-2.1b LoRA std | 〃 시드 7 | 110 |
| 1 | 1 | r2_k8_pair | kanana-1.5-8b QLoRA pair | grad-ckpt, 1 ep 완주, 길이별 묶음, 시드 7 | 300 |
| 1 | 2 | r2_k8_std | kanana-1.5-8b QLoRA std | 〃 시드 13 | 230 |
| 1 | 3 | r2_mdeb_std | mDeBERTa-v3-base std | 5 ep, lr 3e-5 | 18 |
| 1 | 4 | r2_deb_std | DeBERTa-v3-ko std | 5 ep, lr 5e-5 | 16 |

## 0.84와의 차이 (숨기지 않는다)

- R1 val 최고는 단일 0.553, 앙상블 oof 0.563이다. 목표 0.84와 약 0.28 떨어져 있다.
- 0.84는 누수 있는 행 단위 분할에서 나온 값이다. 원 test의 70.4%가 train에도 있는 문장이었고(클래스 균형용 복제 최대 56배), 원 코드를 원 분할로 다시 돌린 앙상블 0.8393에서 train에 같은 문장이 있는 test 행의 정확도는 0.970, 없는 행은 0.561이었다(README, `reports/leakage.md`, `reports/diagnostics/row_split_seen_unseen.md`). 문장 단위 분할에는 이 겹침이 없다.
- GPT-4o 라벨과 독립 재라벨(사람이 아니라 Claude Opus 5.5가 GPT-4o 라벨을 보지 않고 붙인 것)의 일치는 40.1%, κ 0.300이다. 라벨 경계가 크게 흔들려 문장 단위 분할로는 0.84에 닿지 못할 수 있다. 회차마다 진척을 그대로 적고 부풀리지 않는다.

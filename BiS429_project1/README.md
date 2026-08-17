# Project 1 — Chest X-ray Diagnosis (BiS429)

흉부 X-ray 이미지를 **normal / abnormal** 두 클래스로 분류하는 CNN.
학습용 800장(각 클래스 400장), 테스트용 50장(라벨 없음), 이미지는 128×128 grayscale.

## 디렉토리 구성

| 경로 | 설명 |
| --- | --- |
| [data.py](data.py) | 이미지 로딩, train/val 분할, 배치 생성 |
| [model.py](model.py) | `ChestXrayCNN` 모델과 one-hot 교차엔트로피 손실 |
| [train.py](train.py) | 학습 · 평가 · 예측 저장 CLI 진입점 |
| [notebooks/](notebooks/) | 제출 당시의 원본 노트북 (`project1_baseline.ipynb`) |
| [data/](data/) | `Project1_data_files.zip` — 여기에 압축을 풀면 됨 |
| [results/](results/) | 제출한 답안지와 학습 곡선, best run 메모 |
| [docs/](docs/) | 보고서(docx/pdf)와 발표자료(pptx) |

`notebooks/`의 노트북은 제출 기록용 보관본이다. 실행 경로는 `train.py`다.

## 준비

```bash
pip install -r requirements.txt
unzip data/Project1_data_files.zip -d data/
```

압축을 풀면 `data/training_images/`(normal1..400.png, abnormal1..400.png)와
`data/test_images/`(1..50.png)가 만들어진다.

## 실행

```bash
# 기본 설정 (제출 당시와 동일: batch 30, epoch 100, lr 1.5e-4)
python train.py

# 하이퍼파라미터 변경
python train.py --epochs 50 --batch-size 64 --lr 3e-4 --seed 42

# GPU 지정 / 그래프 생략
python train.py --device cuda --no-plot

# 짧게 동작만 확인
python train.py --epochs 2
```

전체 옵션은 `python train.py --help`.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--data-dir` | `data` | `training_images/`, `test_images/`가 있는 디렉토리 |
| `--output-dir` | `results` | 예측 CSV와 그래프 저장 위치 |
| `--epochs` | `100` | 최대 에폭 (조기 종료 조건 충족 시 더 일찍 멈춤) |
| `--batch-size` | `30` | 미니배치 크기 |
| `--lr` | `0.00015` | Adam 초기 학습률 |
| `--lr-decay` | `0.9` | validation 정확도 정체 시 학습률에 곱하는 계수 |
| `--n-train` | `360` | 학습 샘플 수, 나머지 800−n장은 validation |
| `--seed` | `0` | 셔플·초기화 시드 |
| `--device` | 자동 | `cuda` 사용 가능하면 cuda, 아니면 cpu |
| `--no-plot` | off | 학습 곡선 이미지를 만들지 않음 |

## 출력물

- `results/predictions.csv` — 테스트 이미지 1~50번의 예측 (0 = normal, 1 = abnormal)
- `results/learning_curves.png` — epoch별 loss / accuracy
- `results/loss_gap.png` — validation loss − training loss (과적합 관찰용)

## 모델과 학습 규칙

```
입력 (1, 128, 128)
  conv 5×5, 32ch  → ReLU → maxpool 2  →  (32, 64, 64)
  conv 3×3,  4ch  → ReLU → maxpool 2  →  ( 4, 32, 32)
  fc 4096 → 64 → 8 → 2 → softmax
```

- 손실: one-hot 타깃에 대한 교차엔트로피. `forward`가 이미 softmax 확률을 내보내므로
  `nn.CrossEntropyLoss`가 아니라 [model.py](model.py)의 `cross_entropy`를 쓴다.
- 학습률 감쇠: 10 에폭 이후 validation 정확도 변화가 0.1%p 이하이면 `--lr-decay`를 곱한다.
- 조기 종료: training 정확도 ≥ 95% 이면서 train/val 정확도 차이가 3%p 이하일 때 중단.

## 원본 노트북에서 바뀐 점

- 하드코딩된 절대경로(`/home/dhlee/courseworks/...`)를 `--data-dir` 옵션으로 대체.
- 전역 변수(`selected_training_data`, `batch_size`)에 의존하던 배치 함수를 인자를 받는
  제너레이터로 교체.
- 학습률 감쇠가 지역 변수만 바꾸고 optimizer에는 반영되지 않던 문제를 수정 —
  이제 `optimizer.param_groups`의 `lr`을 직접 갱신한다. 즉 노트북 결과와 수치가 다를 수 있다.
- 평가 로직을 `evaluate()` 하나로 합치고, 예측 결과를 CSV로 저장하도록 했다.

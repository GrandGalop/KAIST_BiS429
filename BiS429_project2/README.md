# Project 2 — Predictive Coding on MNIST (BiS429)

역전파 대신 **predictive coding**으로 MNIST 분류기를 학습한다.
각 층은 아래 층의 예측 `mu = W f(x) + b`를 받고, 예측 오차
`eps = (x − mu) / var`가 두 단계를 모두 구동한다.

1. **Inference (E step)** — 가중치를 고정한 채 은닉 상태 `x`를 완화(relaxation)시켜
   자유에너지 `F`를 최대화한다.
2. **Parameter update (M step)** — 수렴한 오차로 가중치와 편향을 Adam 한 스텝 갱신한다.

autograd를 쓰지 않고 모든 gradient를 직접 유도해 구현한 것이 과제의 핵심이라,
학습 전체가 `torch.no_grad()` 안에서 돈다.

## 디렉토리 구성

| 경로 | 설명 |
| --- | --- |
| [functions.py](functions.py) | 활성함수 `f`, 도함수 `f_deriv`, 역함수 `f_inv` |
| [data.py](data.py) | MNIST 로딩, 전처리, 배치 분할, 정확도 계산 |
| [model.py](model.py) | `NetworkForPredictiveCoding` — inference와 parameter update |
| [config.py](config.py) | 기본 하이퍼파라미터(`AttrDict`) |
| [train.py](train.py) | 학습 루프 CLI 진입점 |
| [notebooks/](notebooks/) | 제출 당시의 원본 노트북 (`project2_baseline.ipynb`) |
| [results/](results/) | 100 에폭 학습 로그와 정확도 그래프 |
| [docs/](docs/) | 과제 명세서와 제출 보고서 |

`notebooks/`의 노트북은 제출 기록용 보관본이다. 실행 경로는 `train.py`이며, 노트북을
그대로 돌리려면 상위 디렉토리를 import 경로에 넣어야 한다
(`import sys; sys.path.append("..")`).

데이터 배열은 전부 `(feature, batch)` 방향이다 — 이미지는 `(784, N)`, one-hot 라벨은
`(10, N)`. 갱신식이 전부 왼쪽 곱셈으로 쓰여 있기 때문이다.

## 준비

```bash
pip install -r requirements.txt
```

MNIST는 첫 실행 때 `MNIST/` 아래로 자동 다운로드된다.

## 실행

```bash
# 기본 설정 (제출 당시와 동일: 100 epoch, batch 128, lr 1e-3, [784,500,500,10])
python train.py

# 빠른 동작 확인 — 학습 2000장, 2 에폭
python train.py --data-size 2000 --epochs 2

# 구조·하이퍼파라미터 변경
python train.py --layers 784 300 300 10 --lr 5e-4 --max-iterations 100

# 디바이스 지정 / 재현성 확보
python train.py --device cpu --seed 0
```

전체 옵션은 `python train.py --help`.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--epochs` | `100` | 학습 에폭 수 |
| `--batch-size` | `128` | 미니배치 크기 |
| `--lr` | `1e-3` | Adam 학습률 |
| `--layers` | `784 500 500 10` | 층별 뉴런 수 |
| `--max-iterations` | `50` | 배치마다 수행하는 inference 완화 반복 횟수 |
| `--data-size` | 전체 | 학습셋을 잘라 짧게 돌릴 때 사용 (테스트셋은 그 1/5) |
| `--data-root` | `MNIST` | torchvision MNIST 캐시 위치 |
| `--output-dir` | `results` | 정확도 그래프 저장 위치 |
| `--device` | 자동 | `cuda` 또는 `cpu` |
| `--seed` | 없음 | numpy·torch 시드 |
| `--no-plot` | off | 그래프를 만들지 않음 |

CPU에서 전체 100 에폭은 매우 오래 걸린다. 확인용으로는 `--data-size`를 함께 쓰는 편이 좋다.

## 결과

[results/train_log.txt](results/train_log.txt)는 CUDA에서 100 에폭을 돌린 기록이다.
1 에폭 만에 91%에 도달하고 100 에폭에서 test 정확도 **96.2%**로 끝난다.
epoch별 곡선은 [results/accuracy_per_epoch.png](results/accuracy_per_epoch.png).

## 구현 노트

- **자유에너지 `F`** 는 오차 제곱의 단순 합이 아니라 분산으로 가중된
  `−½ Σ_l (x_l − mu_l)² / var_l` 이다.
- **완화 스텝 크기** `beta`는 `F`가 감소하면(오버슛) 절반으로 줄인다. 변화량이
  임계값 아래로 내려가면 `--max-iterations` 전에 조기 종료한다.
- **부호** — M step은 `F`를 *최대화*하므로 갱신식이 `W + lr · …` 형태다.

## 원본 노트북에서 바뀐 점

- 노트북 셀 하나에 뭉쳐 있던 모델·데이터·설정을 모듈 네 개로 분리하고 CLI를 붙였다.
- `torch.ones(...).cuda()` 하드코딩을 제거해 CPU에서도 돌아간다. bias gradient는
  동일한 값을 주는 `errors.sum(dim=1, keepdim=True)`로 바꿨다.
- 활성함수 상수 비교를 `is`에서 `==`로 바꿨다. 문자열 identity 비교는 보장되지 않는다.
- `f_deriv(LINEAR)`가 torch 텐서 입력에 numpy 배열을 돌려주던 것을 `torch.ones_like`로 수정.
- 오차·자유에너지를 세 군데에서 중복 계산하던 코드를 `_free_energy_and_errors` 하나로 합쳤다.
- 오타 `type_of_optimzer` → `type_of_optimizer`, `_parameter_intialization` → `_initialize_parameters`.
- bias gradient에 `1/size_of_batch`가 두 번 걸리는 부분은 제출 당시 수치를 보존하려고
  **그대로 두었다** ([model.py](model.py)에 주석으로 표시).

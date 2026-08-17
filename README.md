# BiS429 (뇌기반 기계지능) Term Projects

KAIST BiS429 기말 프로젝트 두 개. 각 디렉토리에 실행 방법을 담은 README가 따로 있다.

| 프로젝트 | 주제 | 실행 |
| --- | --- | --- |
| [BiS429_project1](BiS429_project1/) | 흉부 X-ray 이미지 정상/비정상 이진 분류 CNN | `cd BiS429_project1 && python train.py` |
| [BiS429_project2](BiS429_project2/) | Predictive coding 학습 알고리즘으로 MNIST 분류 | `cd BiS429_project2 && python train.py` |

- 자세한 옵션과 준비 과정: [Project 1 README](BiS429_project1/README.md),
  [Project 2 README](BiS429_project2/README.md)
- 두 프로젝트 모두 각자의 `requirements.txt`를 가진다.
- 각 디렉토리는 같은 구조를 따른다 — 최상위에 실행 가능한 모듈, 그리고
  `notebooks/`(제출 당시 원본), `results/`(로그·그림·답안), `docs/`(보고서·명세서).

---

## Project 1 — Medical Image Diagnosis

- **목표:** 흉부 X-ray 이미지를 **normal vs abnormal** 로 분류.
- **데이터:** 128×128 grayscale. 라벨된 학습·검증용 800장, 라벨 없는 테스트 50장.
- **절차:** 픽셀값 정규화 → CNN 학습 → validation으로 하이퍼파라미터 조정 →
  테스트 50장 예측 후 답안지 제출.
- **구현:** conv(32ch) → conv(4ch) → fc 64 → 8 → 2 → softmax. one-hot 교차엔트로피,
  Adam, validation 정확도 정체 시 학습률 감쇠, train/val 격차 기준 조기 종료.

## Project 2 — Predictive Coding

역전파 없이, 각 뉴런이 아래 층의 예측 오차만 보고 학습하는 predictive coding 네트워크를
구현하고 MNIST로 학습한다. Expectation-Maximization 형태의 두 단계를 번갈아 수행한다.

- **Inference (E step):** 가중치를 고정한 채 뉴런 상태 `x`를 갱신해 자유에너지 `F`를 최대화.
- **Parameter update (M step):** 수렴한 오차로 가중치·편향을 갱신해 예측 오차를 줄임.

### 핵심 수식

- 예측: $\mu_l = W_{l-1} f(x_{l-1}) + b_{l-1}$
- 예측 오차: $\epsilon_l = (x_l - \mu_l) / \sigma_l^2$
- 자유에너지: $F = -\frac{1}{2}\sum_l (x_l - \mu_l)^2 / \sigma_l^2$
- 파라미터 기울기:
  $\frac{\partial F}{\partial W_{l-1}} = \epsilon_l f(x_{l-1})^\top$,
  $\frac{\partial F}{\partial b_{l-1}} = \sum_{batch} \epsilon_l$

**결과:** MNIST 테스트 정확도 1 에폭 91%, 100 에폭 **96.2%**
(로그: [BiS429_project2/results/train_log.txt](BiS429_project2/results/train_log.txt)).

### 생물학적 타당성

가중치 갱신에 필요한 정보가 인접한 두 층의 국소 신호(`ε_l`, `f(x_{l-1})`)뿐이므로,
전역 역전파 경로를 가정하지 않아도 학습이 성립한다.

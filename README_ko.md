# Spring-Mass 역추정 실습

NVIDIA Warp를 사용해 2차원 spring-mass 시뮬레이션과 스프링 강성 역추정을 실습하는 프로젝트입니다.

## Tutorial 01

입자들을 구조 스프링과 전단 스프링으로 연결해 격자를 구성합니다. 상단 행은 고정하고, 하단 중앙 부근에 일정 시간 아래 방향 힘을 가합니다. 감쇠와 semi-implicit Euler 적분을 사용합니다.

그 후 Warp autodiff와 Adam 최적화를 이용해 관측 궤적에서 스프링 강성 `k`를 추정합니다.

## 설치

```bash
pip install -r requirements.txt
```

GPU는 필수가 아닙니다. 사용 가능한 CUDA GPU가 있으면 CUDA를 사용하고, 없으면 CPU로 실행됩니다.

## 실행

```bash
python scripts/tutorial_01_spring_mass.py
```

그래프 창은 열리지 않으며 결과는 다음 위치에 저장됩니다.

```text
outputs/tutorial_01/
```

애니메이션 GIF, 최적화 과정, 궤적 비교 그래프가 생성됩니다.

## Tutorial 02

Tutorial 02는 3차원 spring-mass sheet로 확장한 예제입니다. 최신 Warp 기반 물리 엔진인 Newton을 사용해 중력, 바닥 충돌, 반발을 처리합니다.

PhysTwin의 축소 버전처럼 모든 spring을 각각 최적화하지 않고, 하나의 global stiffness scale과 왼쪽·중앙·오른쪽 세 영역의 stiffness scale을 최적화합니다. `Initial Guess`, `Optimized`, `Ground Truth` 결과를 3D GIF로 비교합니다.

```bash
python scripts/tutorial_02_gravity_collision.py
```

결과는 `outputs/tutorial_02/`에 GIF와 최종 상태 PNG로 저장됩니다.

## Tutorial 03

Tutorial 03은 grasp particle 위치를 `p0`, `p5`로 알고 있다고 가정하고, 두 점에 필요한 시간별 외력을 Warp 자동미분과 Adam으로 추정합니다. 시뮬레이션은 36개 particle 전체를 계산하며 loss는 trajectory loss만 사용합니다. GIF에서 Target, 초기 외력(0 N), 최적화 결과를 비교합니다.

```bash
python scripts/tutorial_03_sparse_force_inference.py
```

결과는 `outputs/tutorial_03/`에 저장됩니다.

```text
cloth_trajectory_comparison.gif
trajectory_and_force.png
target_trajectory.npy
ground_truth_force_knots.npy
initial_force_knots.npy
optimized_force_knots.npy
```

힘 배열은 `(11, 36, 3)` 형태이며 단위는 N입니다. 순서는 시간 knot, particle, XYZ 성분이며 knot 간격은 0.032초입니다.

## Tutorial 03: 수직 grasp 힘 추정

`tutorial_03_vertical_grasp_force_inference.py`는 GT 궤적을 만들 때 grasp particle `p0`, `p5`의 X/Y 위치를 유지합니다. 힘 추정 단계에서는 grasp 점을 자유롭게 두고, Adam이 GT 궤적을 따라가도록 두 점의 XYZ 힘을 모두 0N에서 최적화합니다. 힘 비교 그래프에서 GT, 초기값, 최적화 결과의 `Fx`, `Fy`, `Fz`를 확인할 수 있습니다.

```bash
python scripts/tutorial_03_vertical_grasp_force_inference.py
```

결과는 `outputs/tutorial_03_1/`에 저장됩니다.

```text
vertical_grasp_force_comparison.gif
trajectory_and_force_comparison.png
target_trajectory.npy
target_force_knots.npy
target_grasp_reaction_forces.npy
target_grasp_forces_xyz.npy
initial_force_knots.npy
optimized_force_knots.npy
```

힘 knot 배열은 `(11, 2, 3)` 형태이며 단위는 N입니다(시간 knot, grasp particle, XYZ 성분). knot 간격은 0.032초입니다. `target_grasp_forces_xyz.npy`에는 매 simulation step의 전체 GT XYZ 힘이 `(80, 2, 3)` 형태로 저장됩니다.

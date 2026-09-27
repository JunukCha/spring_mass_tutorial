# Tutorial 스크립트 안내

이 폴더에는 spring-mass 실습 코드가 있습니다.

## Tutorial 01

`tutorial_01_spring_mass.py`는 다음 내용을 다룹니다.

- 2차원 spring-mass 격자 구성
- 구조 스프링과 전단 스프링
- 상단 경계조건
- Warp 기반 시뮬레이션
- spring stiffness `k` 역추정
- Adam과 자동미분

실행:

```bash
python scripts/tutorial_01_spring_mass.py
```

## Tutorial 02

`tutorial_02_gravity_collision.py`는 3차원 spring-mass sheet를 사용합니다.

- Newton 기반 물리 시뮬레이션
- Z축 방향 중력
- XY 바닥 평면 충돌
- 충돌 반발계수
- global stiffness와 region별 stiffness 최적화
- Initial Guess / Optimized / Ground Truth 비교

실행:

```bash
python scripts/tutorial_02_gravity_collision.py
```

## Tutorial 02의 파라미터 구조

모든 spring에 서로 다른 stiffness를 주지 않고, 기준 stiffness와 영역별 scale을 사용합니다.

```text
1단계: global_scale 최적화
2단계: region_scale을 global_scale로 초기화
최종: k_spring = BASE_STIFFNESS × region_scale
```

현재 Ground Truth는 실제 지역별 `k` 값으로 정의하고, damping은 고정되어 있습니다.

## 출력 위치

결과는 다음 폴더에 저장됩니다.

```text
outputs/tutorial_01/
outputs/tutorial_02/
```

Tutorial 02는 다음 결과를 저장합니다.

```text
spring_mass_3d_comparison.gif
spring_mass_3d_final_states.png
```

## Tutorial 03

`tutorial_03_sparse_force_inference.py`는 grasp 위치를 알고 있다고 가정하고(`p0`, `p5`), 두 particle에 필요한 시간별 XYZ 외력을 Warp 자동미분과 Adam으로 추정합니다. GIF에서 목표, 초기 외력(0 N), 최적화된 천의 궤적을 비교합니다.

프로젝트 루트에서 실행:

```bash
python scripts/tutorial_03_sparse_force_inference.py
```

결과는 `outputs/tutorial_03/`에 저장됩니다. 힘 `.npy` 배열은 `(11, 36, 3)` 형태이며 단위는 N입니다(시간 knot, particle, XYZ). knot 간격은 0.032초입니다. 목표 힘, 초기 힘, 최적화된 힘과 목표 particle trajectory를 저장합니다.

## Tutorial 03: 수직 grasp 힘 추정

`tutorial_03_vertical_grasp_force_inference.py`는 grasp particle `p0`, `p5`를 수직 가이드로 들어 올려 GT 궤적을 만듭니다. 힘 추정에서는 grasp 점을 자유롭게 두고 Warp 자동미분과 Adam으로 `Fx`, `Fy`, `Fz`를 모두 0N부터 최적화합니다. 목적 함수는 trajectory loss입니다.

프로젝트 루트에서 실행:

```bash
python scripts/tutorial_03_vertical_grasp_force_inference.py
```

결과는 `outputs/tutorial_03_1/`에 저장됩니다. GIF는 목표, 0N 초기 힘, 최적화된 궤적을 비교하며 `trajectory_and_force_comparison.png`는 GT와 최적화된 `Fx`, `Fy`, `Fz` knot를 함께 그립니다. 힘 knot 배열은 `(11, 2, 3)` 형태이며 단위는 N입니다(시간 knot, grasp particle, XYZ). knot 간격은 0.032초입니다. 매 step의 전체 GT 힘은 `target_grasp_forces_xyz.npy`에 `(80, 2, 3)` 형태로 저장됩니다. GT 반력, 목표 궤적, 초기 힘, 최적화된 힘 배열도 함께 저장합니다.

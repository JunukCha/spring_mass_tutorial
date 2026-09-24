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
k_spring = BASE_STIFFNESS × global_scale × region_scale
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

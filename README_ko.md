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

# Tutorial 01 설명

`tutorial_01_spring_mass.py`는 다음 과정을 하나의 실습으로 보여줍니다.

1. 2차원 입자 격자 생성
2. 구조 스프링과 전단 스프링 연결
3. 상단 입자 고정
4. 하단 중앙 입자에 임시 외력 적용
5. 스프링 힘과 감쇠력 계산
6. semi-implicit Euler 방식으로 시간 적분
7. Warp autodiff로 스프링 강성 `k`의 gradient 계산
8. Adam으로 `k` 역추정
9. 결과를 GIF와 PNG로 저장

## 실행

프로젝트 루트에서 실행합니다.

```bash
python scripts/tutorial_01_spring_mass.py
```

결과는 `outputs/tutorial_01/`에 저장됩니다.

## 주요 함수

- `particle_id()` — 격자 좌표를 입자 인덱스로 변환
- `simulate_step()` — 한 타임스텝의 물리 계산 Warp 커널
- `record_positions()` — 궤적 기록 Warp 커널
- `compute_loss()` — 예측 궤적과 관측 궤적의 loss 계산
- `simulate()` — 전체 시뮬레이션 실행
- `trajectory_to_numpy()` — Warp 배열을 NumPy 궤적으로 변환
- `draw_object()` — 입자와 스프링 시각화
- `main()` — 관측 생성, 최적화, 결과 저장을 실행

현재 모델에는 중력이 없고, `PUSH_FORCE`로 지정한 임시 아래 방향 외력만 적용됩니다.

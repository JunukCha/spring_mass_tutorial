# Spring-Mass Inverse Modeling

This project demonstrates spring-mass simulation and inverse modeling with NVIDIA Warp and Newton.

## Tutorial 01

The tutorial creates a grid of particles connected by structural and shear springs. The top row is fixed, and a temporary downward force is applied near the center of the bottom row. The simulation uses damping and semi-implicit Euler integration.

The inverse modeling step estimates the spring stiffness `k` from a ground-truth trajectory using Warp autodiff and Adam optimization.

## Requirements

```bash
pip install -r requirements.txt
```

GPU is optional. Warp automatically uses CUDA when available and otherwise falls back to CPU.

## Run

```bash
python scripts/tutorial_01_spring_mass.py
```

The script does not open plot windows. Results are saved to:

```text
outputs/tutorial_01/
```

Generated files include the animation GIF, optimization history, and trajectory comparison plot.

See [README_ko.md](README_ko.md) for the Korean documentation.

## Tutorial 02

Tutorial 02 extends the model to a 3D spring-mass sheet and uses Newton, the modern physics engine built on NVIDIA Warp. It includes gravity, ground contact, restitution, and a reduced PhysTwin-style inverse fitting workflow.

Instead of optimizing every spring independently, the model uses one global stiffness scale and three regional stiffness scales. The Ground Truth, Initial Guess, and Optimized trajectories are compared in a 3D animation.

```bash
python scripts/tutorial_02_gravity_collision.py
```

Results are saved to `outputs/tutorial_02/` as a GIF and a final-state PNG.

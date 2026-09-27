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

## Tutorial 03

Tutorial 03 estimates time-varying external forces on two known grasp particles (`p0` and `p5`) with Warp autodiff and Adam. The simulation includes all 36 particle trajectories, while the objective uses trajectory loss only. The GIF compares the target, zero-force initial state, and optimized result.

```bash
python scripts/tutorial_03_sparse_force_inference.py
```

Results are saved to `outputs/tutorial_03/`:

```text
cloth_trajectory_comparison.gif
trajectory_and_force.png
target_trajectory.npy
ground_truth_force_knots.npy
initial_force_knots.npy
optimized_force_knots.npy
```

The force arrays have shape `(11, 36, 3)` in newtons, indexed by time knot, particle, and XYZ component. Knots are 0.032 seconds apart.

## Tutorial 03: vertical grasp force inference

`tutorial_03_vertical_grasp_force_inference.py` generates a target lift while keeping grasp particles `p0` and `p5` at their original X/Y positions. During inference, the grasp particles are free and Adam optimizes their full XYZ forces from zero initialization to match the target trajectory. The force comparison plot shows target, initial, and optimized `Fx`, `Fy`, and `Fz`.

```bash
python scripts/tutorial_03_vertical_grasp_force_inference.py
```

Results are saved to `outputs/tutorial_03_1/`:

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

Force knot arrays have shape `(11, 2, 3)` in newtons (time knot, grasp particle, XYZ); knots are 0.032 seconds apart. `target_grasp_forces_xyz.npy` stores the full GT XYZ force at every simulation step with shape `(80, 2, 3)`.

# Tutorial Scripts

This directory contains the spring-mass tutorial scripts.

## Tutorial 01

`tutorial_01_spring_mass.py` covers:

- 2D spring-mass grid construction
- Structural and shear springs
- Fixed boundary conditions
- Warp-based simulation
- Spring stiffness `k` estimation
- Automatic differentiation and Adam optimization

Run it from the project root:

```bash
python scripts/tutorial_01_spring_mass.py
```

## Tutorial 02

`tutorial_02_gravity_collision.py` uses a 3D spring-mass sheet.

- Newton-based physics simulation
- Gravity along the Z axis
- Ground collision on the XY plane
- Collision restitution
- Global and regional stiffness optimization
- Initial Guess / Optimized / Ground Truth comparison

Run it from the project root:

```bash
python scripts/tutorial_02_gravity_collision.py
```

## Tutorial 03

`tutorial_03_sparse_force_inference.py` fits time-varying XYZ forces at two known grasp particles (`p0` and `p5`) using Warp autodiff and Adam. It compares target, zero-force initial, and optimized cloth trajectories in a GIF.

Run from the project root:

```bash
python scripts/tutorial_03_sparse_force_inference.py
```

Outputs are written to `outputs/tutorial_03/`. The `.npy` force arrays are shaped `(11, 36, 3)` in newtons (time knot, particle, XYZ); knots are 0.032 seconds apart. Files include target, initial, and optimized force controls, plus the target particle trajectory.

## Tutorial 03: vertical grasp force inference

`tutorial_03_vertical_grasp_force_inference.py` creates a vertically guided target lift for grasp particles `p0` and `p5`. The inference rollout leaves those particles free and uses Warp autodiff with Adam to fit all three force components, `Fx`, `Fy`, and `Fz`, from zero initialization using trajectory loss.

Run from the project root:

```bash
python scripts/tutorial_03_vertical_grasp_force_inference.py
```

Outputs are written to `outputs/tutorial_03_1/`. The GIF compares target, zero-force initial, and optimized cloth trajectories. `trajectory_and_force_comparison.png` plots target and optimized `Fx`, `Fy`, and `Fz` knots together. Force knot arrays have shape `(11, 2, 3)` in newtons (time knot, grasp particle, XYZ); knots are 0.032 seconds apart. The full per-step target force is saved as `target_grasp_forces_xyz.npy` with shape `(80, 2, 3)`. Other arrays include target reaction forces, the target trajectory, and initial and optimized force knots.

## Tutorial 02 parameterization

The model does not assign an independent stiffness to every spring. It uses a base stiffness and regional scale parameters:

```text
Stage 1: estimate global_scale
Stage 2: initialize region_scale from global_scale
k_spring = BASE_STIFFNESS × region_scale
```

Ground Truth is defined using physical regional `k` values, while damping remains fixed.

## Outputs

Results are saved to:

```text
outputs/tutorial_01/
outputs/tutorial_02/
```

Tutorial 02 generates:

```text
spring_mass_3d_comparison.gif
spring_mass_3d_final_states.png
```

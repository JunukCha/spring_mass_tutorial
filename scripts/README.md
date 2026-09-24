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

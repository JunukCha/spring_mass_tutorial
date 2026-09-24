"""Tutorial 02: reduced PhysTwin-style sparse stiffness fitting.

A 3D spring-mass sheet is simulated with Newton. The material is represented
by three regional stiffness scales instead of one stiffness per spring.
The regional scales are recovered from a synthetic ground-truth trajectory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import newton
import numpy as np
from scipy.optimize import minimize
import warp as wp
from newton.solvers import SolverXPBD


ROWS = 6
COLS = 6
SPACING = 0.20
PARTICLE_MASS = 1.0
PARTICLE_RADIUS = 0.045

BASE_STIFFNESS = 500.0
SPRING_DAMPING = 2.5

GRAVITY = 9.81
GROUND_Z = -(ROWS - 1) * SPACING - 0.5
GROUND_RESTITUTION = 0.65
GROUND_FRICTION = 0.4
IMPACT_VELOCITY = -2.0

NUM_STEPS = 1000
OPTIMIZATION_STEPS = 300
DT = 0.002
FRAME_STRIDE = 4
OPTIMIZATION_STRIDE = 3
GIF_FPS = 30

OUTPUT_DIR = Path("outputs") / "tutorial_02"

GROUND_TRUTH_K = np.array([595.0, 700.0, 805.0])
INITIAL_SCALES = np.array([0.55, 1.35, 0.65])


def particle_id(row, col):
    return row * COLS + col


def region_id_from_particle(particle_index):
    """Assign a spring to left, center, or right using its x location."""

    col = particle_index % COLS
    if col < COLS // 3:
        return 0
    if col < 2 * COLS // 3:
        return 1
    return 2


def spring_stiffness(i, j, region_scales):
    region_i = region_id_from_particle(i)
    region_j = region_id_from_particle(j)
    region = round((region_i + region_j) / 2)
    region = min(region, len(region_scales) - 1)
    return BASE_STIFFNESS * float(region_scales[region])


def build_model(region_scales, device):
    builder = newton.ModelBuilder(
        up_axis="Z",
        gravity=(0.0, 0.0, -GRAVITY),
    )

    for row in range(ROWS):
        for col in range(COLS):
            builder.add_particle(
                pos=(col * SPACING, 0.0, -row * SPACING),
                vel=(
                    0.0,
                    0.0,
                    IMPACT_VELOCITY if row == ROWS - 1 else 0.0,
                ),
                mass=PARTICLE_MASS,
                radius=PARTICLE_RADIUS,
            )

    def add_spring(i, j):
        builder.add_spring(
            i,
            j,
            spring_stiffness(i, j, region_scales),
            SPRING_DAMPING,
            0.0,
        )

    for row in range(ROWS):
        for col in range(COLS):
            current = particle_id(row, col)
            if col < COLS - 1:
                add_spring(current, particle_id(row, col + 1))
            if row < ROWS - 1:
                add_spring(current, particle_id(row + 1, col))

    for row in range(ROWS - 1):
        for col in range(COLS - 1):
            add_spring(
                particle_id(row, col),
                particle_id(row + 1, col + 1),
            )
            add_spring(
                particle_id(row, col + 1),
                particle_id(row + 1, col),
            )

    ground_config = builder.ShapeConfig(
        ke=100000.0,
        kd=100.0,
        mu=GROUND_FRICTION,
        restitution=GROUND_RESTITUTION,
    )
    builder.add_ground_plane(height=GROUND_Z, cfg=ground_config)
    return builder.finalize(device=device)


def simulate(region_scales, device, num_steps):
    model = build_model(region_scales, device)
    solver = SolverXPBD(model, enable_restitution=True)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    state = model.state()
    next_state = model.state()
    control = model.control()
    positions = [state.particle_q.numpy().copy()]

    for _ in range(num_steps):
        state.clear_forces()
        collision_pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, DT)
        state, next_state = next_state, state
        positions.append(state.particle_q.numpy().copy())

    return np.asarray(positions)


def simulate_ground_truth(device, num_steps):
    return simulate(GROUND_TRUTH_K / BASE_STIFFNESS, device, num_steps)


def optimize_global_scale(target, device):
    def loss(log_scale):
        scale = float(np.exp(log_scale[0]))
        prediction = simulate(
            np.full(3, scale),
            device,
            OPTIMIZATION_STEPS,
        )
        error = prediction[::OPTIMIZATION_STRIDE] - target[::OPTIMIZATION_STRIDE]
        return float(np.mean(error * error))

    result = minimize(
        loss,
        np.array([1.0]),
        method="Powell",
        bounds=[(-2.0, 2.0)],
        options={"maxiter": 10, "xtol": 1.0e-2, "ftol": 1.0e-8},
    )
    return float(np.exp(result.x[0])), float(result.fun)


def optimize_region_scales(target, initial_global_scale, device):
    def loss(log_scales):
        scales = np.exp(log_scales)
        prediction = simulate(scales, device, OPTIMIZATION_STEPS)
        error = prediction[::OPTIMIZATION_STRIDE] - target[::OPTIMIZATION_STRIDE]
        return float(np.mean(error * error))

    result = minimize(
        loss,
        np.log(np.full(3, initial_global_scale)),
        method="Powell",
        bounds=[(-2.0, 2.0)] * 3,
        options={"maxiter": 20, "xtol": 1.0e-2, "ftol": 1.0e-8},
    )
    return np.exp(result.x), float(result.fun)


def draw_sheet(ax, positions, title, color):
    ax.clear()

    for row in range(ROWS):
        for col in range(COLS):
            current = particle_id(row, col)

            if col < COLS - 1:
                other = particle_id(row, col + 1)
                ax.plot(
                    [positions[current, 0], positions[other, 0]],
                    [positions[current, 1], positions[other, 1]],
                    [positions[current, 2], positions[other, 2]],
                    color=color,
                    linewidth=1.0,
                )

            if row < ROWS - 1:
                other = particle_id(row + 1, col)
                ax.plot(
                    [positions[current, 0], positions[other, 0]],
                    [positions[current, 1], positions[other, 1]],
                    [positions[current, 2], positions[other, 2]],
                    color=color,
                    linewidth=1.0,
                )

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color=color,
        s=22,
    )

    ground_x = np.linspace(-0.15, (COLS - 1) * SPACING + 0.15, 2)
    ground_y = np.linspace(-0.45, 0.45, 2)
    ground_x, ground_y = np.meshgrid(ground_x, ground_y)
    ground_z = np.full_like(ground_x, GROUND_Z)
    ax.plot_surface(
        ground_x,
        ground_y,
        ground_z,
        alpha=0.38,
        color="lightgray",
        edgecolor="none",
    )

    ax.set_title(title)
    ax.set_xlim(-0.15, (COLS - 1) * SPACING + 0.15)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(GROUND_Z - 0.1, 0.15)
    ax.set_box_aspect((1.2, 1.2, 0.8))
    ax.set_xlabel("X")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (height)")
    ax.set_proj_type("ortho")
    ax.view_init(elev=25.0, azim=-65.0)


def save_comparison(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = ["Initial Guess", "Optimized", "Ground Truth"]

    fig = plt.figure(figsize=(18, 6))
    axes = [
        fig.add_subplot(1, 3, index + 1, projection="3d")
        for index in range(3)
    ]

    colors = ["tab:blue", "tab:orange", "tab:green"]

    def update(frame):
        fig.suptitle(
            f"Tutorial 02: Reduced PhysTwin | t = {frame * DT:.3f} s",
            fontsize=16,
            y=0.98,
        )

        for ax, name, color in zip(axes, names, colors):
            draw_sheet(ax, results[name][frame], name, color)

        return axes

    animation = FuncAnimation(
        fig,
        update,
        frames=range(0, NUM_STEPS + 1, FRAME_STRIDE),
        interval=40,
        repeat=False,
    )
    fig.subplots_adjust(
        top=0.84,
        bottom=0.08,
        left=0.02,
        right=0.98,
        wspace=0.12,
    )
    animation.save(
        OUTPUT_DIR / "spring_mass_3d_comparison.gif",
        writer="pillow",
        fps=GIF_FPS,
    )
    plt.close(fig)

    final_fig = plt.figure(figsize=(18, 6))
    for index, name in enumerate(names):
        ax = final_fig.add_subplot(1, 3, index + 1, projection="3d")
        draw_sheet(ax, results[name][-1], name, colors[index])

    final_fig.suptitle("Tutorial 02: Final 3D States", fontsize=16)
    final_fig.subplots_adjust(
        top=0.84,
        bottom=0.08,
        left=0.02,
        right=0.98,
        wspace=0.12,
    )
    final_fig.savefig(
        OUTPUT_DIR / "spring_mass_3d_final_states.png",
        dpi=150,
    )
    plt.close(final_fig)


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    print(f"Using device: {device}")

    print("Simulating Ground Truth...")
    optimization_target = simulate_ground_truth(
        device,
        OPTIMIZATION_STEPS,
    )

    print("Optimizing global stiffness scale...")
    optimized_global, global_loss = optimize_global_scale(
        optimization_target,
        device,
    )

    print("Optimizing regional stiffness scales...")
    optimized_regions, final_loss = optimize_region_scales(
        optimization_target,
        optimized_global,
        device,
    )

    ground_truth = simulate_ground_truth(device, NUM_STEPS)

    print("Simulating Initial Guess...")
    initial = simulate(INITIAL_SCALES, device, NUM_STEPS)

    print("Simulating Optimized...")
    optimized = simulate(optimized_regions, device, NUM_STEPS)

    initial_k = BASE_STIFFNESS * INITIAL_SCALES
    optimized_k = BASE_STIFFNESS * optimized_regions

    print(f"Ground Truth k: {GROUND_TRUTH_K}")
    print(f"Initial k:      {initial_k}")
    print(f"Optimized k:    {optimized_k}")
    print(f"Optimized global scale:  {optimized_global:.6f}")
    print(f"Optimized region scales: {optimized_regions}")
    print(f"Global-stage loss: {global_loss:.8e}")
    print(f"Final trajectory loss: {final_loss:.8e}")

    save_comparison(
        {
            "Initial Guess": initial,
            "Optimized": optimized,
            "Ground Truth": ground_truth,
        },
    )
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


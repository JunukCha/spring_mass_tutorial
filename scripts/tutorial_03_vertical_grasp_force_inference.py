"""Tutorial 03: infer 3D grasp forces for a vertical cloth lift.

The target is generated with vertically guided grasp points. During force
inference, grasp points are free and Adam optimizes XYZ force controls to match
the target trajectory using Warp autodiff.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import warp as wp


ROWS = COLS = 6
NUM_PARTICLES = ROWS * COLS
GRASP_0 = 0
GRASP_1 = COLS - 1
GRASP_PARTICLES = (GRASP_0, GRASP_1)
SPACING = 0.20
PARTICLE_MASS = 0.005
SPRING_STIFFNESS = 20.0
SPRING_DAMPING = 0.1
GRAVITY = 9.81
GROUND_Z = 0.0
GROUND_STIFFNESS = 20.0
GROUND_DAMPING = 0.63
PARTICLE_RADIUS = 0.045
DT = 0.004
NUM_STEPS = 80
CONTROL_STRIDE = 8
ADAM_ITERATIONS = 300
ADAM_LEARNING_RATE = 0.05
OUTPUT_DIR = Path("outputs") / "tutorial_03_1"


@wp.kernel
def guided_cloth_step(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_initial: wp.array(dtype=wp.vec3),
    force_knots: wp.array2d(dtype=wp.vec3),
    grasp_reactions: wp.array(dtype=wp.vec3),
    q_next: wp.array(dtype=wp.vec3),
    qd_next: wp.array(dtype=wp.vec3),
    step: int,
    num_knots: int,
    control_stride: int,
    guide_grasps: bool,
    dt: float,
):
    i = wp.tid()
    row = i // COLS
    col = i - row * COLS
    position = q[i]
    velocity = qd[i]
    force = wp.vec3(0.0, 0.0, -PARTICLE_MASS * GRAVITY)

    # Stretch and diagonal shear springs; each particle sums its neighbors.
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr = row + dr
            nc = col + dc
            if (dr != 0 or dc != 0) and nr >= 0 and nr < ROWS and nc >= 0 and nc < COLS:
                j = nr * COLS + nc
                delta = q[j] - position
                distance = wp.length(delta)
                rest_length = SPACING
                if dr != 0 and dc != 0:
                    rest_length = SPACING * 1.41421356237
                direction = delta / wp.max(distance, 1.0e-6)
                relative_speed = wp.dot(qd[j] - velocity, direction)
                force = force + (
                    SPRING_STIFFNESS * (distance - rest_length)
                    + SPRING_DAMPING * relative_speed
                ) * direction

    # Interpolate the two 3D force controls between temporal knots.
    knot = step // control_stride
    alpha = float(step - knot * control_stride) / float(control_stride)
    if knot >= num_knots - 1:
        knot = num_knots - 1
        next_knot = knot
        alpha = 0.0
    else:
        next_knot = knot + 1
    if i == GRASP_0:
        force = force + force_knots[knot, 0] * (1.0 - alpha) + force_knots[next_knot, 0] * alpha
    elif i == GRASP_1:
        force = force + force_knots[knot, 1] * (1.0 - alpha) + force_knots[next_knot, 1] * alpha

    # Record the horizontal force supplied by ideal guides in target generation.
    if guide_grasps:
        if i == GRASP_0:
            grasp_reactions[step * 2] = wp.vec3(-force[0], -force[1], 0.0)
        elif i == GRASP_1:
            grasp_reactions[step * 2 + 1] = wp.vec3(-force[0], -force[1], 0.0)

    # Differentiable penalty contact with the horizontal floor.
    floor_height = GROUND_Z + PARTICLE_RADIUS
    if position[2] < floor_height:
        penetration = floor_height - position[2]
        downward_speed = wp.min(velocity[2], 0.0)
        force[2] = force[2] + GROUND_STIFFNESS * penetration - GROUND_DAMPING * downward_speed

    velocity_next = velocity + dt * force / PARTICLE_MASS
    position_next = position + dt * velocity_next

    # Target generation uses ideal vertical guides. Inference leaves grasp
    # particles free so the optimizer must discover the required XYZ forces.
    if guide_grasps and (i == GRASP_0 or i == GRASP_1):
        position_next = wp.vec3(q_initial[i][0], q_initial[i][1], position_next[2])
        velocity_next = wp.vec3(0.0, 0.0, velocity_next[2])
    elif not guide_grasps and (i == GRASP_0 or i == GRASP_1):
        if i == GRASP_0:
            grasp_reactions[step * 2] = wp.vec3(0.0, 0.0, 0.0)
        else:
            grasp_reactions[step * 2 + 1] = wp.vec3(0.0, 0.0, 0.0)

    q_next[i] = position_next
    qd_next[i] = velocity_next


@wp.kernel
def trajectory_error_step(
    q: wp.array(dtype=wp.vec3),
    target: wp.array2d(dtype=wp.vec3),
    terms: wp.array2d(dtype=float),
    time_index: int,
):
    i = wp.tid()
    delta = q[i] - target[time_index, i]
    terms[time_index, i] = wp.dot(delta, delta)


@wp.kernel
def reduce_trajectory_loss(
    terms: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
    num_times: int,
):
    if wp.tid() == 0:
        total = float(0.0)
        for t in range(num_times):
            for i in range(NUM_PARTICLES):
                total = total + terms[t, i]
        loss[0] = total / float(num_times * NUM_PARTICLES * 3)


def knot_steps():
    result = list(range(0, NUM_STEPS + 1, CONTROL_STRIDE))
    if result[-1] != NUM_STEPS:
        result.append(NUM_STEPS)
    return result


def initial_positions():
    return np.asarray(
        [(col * SPACING, row * SPACING, GROUND_Z + PARTICLE_RADIUS)
         for row in range(ROWS) for col in range(COLS)],
        dtype=np.float32,
    )


def simulate(force_controls, device, target=None, differentiate=False, guide_grasps=False):
    controls = wp.array(
        force_controls, dtype=wp.vec3, device=device, requires_grad=differentiate
    )
    q0 = wp.array(
        initial_positions(), dtype=wp.vec3, device=device, requires_grad=differentiate
    )
    qd0 = wp.zeros(NUM_PARTICLES, dtype=wp.vec3, device=device, requires_grad=differentiate)
    grasp_reactions = wp.empty(NUM_STEPS * 2, dtype=wp.vec3, device=device)
    q_states = [q0]
    qd_states = [qd0]
    for _ in range(NUM_STEPS):
        q_states.append(wp.empty(NUM_PARTICLES, dtype=wp.vec3, device=device, requires_grad=differentiate))
        qd_states.append(wp.empty(NUM_PARTICLES, dtype=wp.vec3, device=device, requires_grad=differentiate))

    if target is not None:
        target_array = wp.array(target, dtype=wp.vec3, device=device)
        terms = wp.empty((NUM_STEPS + 1, NUM_PARTICLES), dtype=float,
                         device=device, requires_grad=differentiate)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=differentiate)
    else:
        target_array = terms = loss = None

    tape = wp.Tape() if differentiate else None
    context = tape if differentiate else _NullContext()
    with context:
        for step in range(NUM_STEPS):
            wp.launch(
                guided_cloth_step,
                dim=NUM_PARTICLES,
                inputs=[q_states[step], qd_states[step], q0, controls, grasp_reactions,
                        q_states[step + 1], qd_states[step + 1], step,
                        len(force_controls), CONTROL_STRIDE, guide_grasps, DT],
                device=device,
            )
        if target_array is not None:
            for t in range(NUM_STEPS + 1):
                wp.launch(
                    trajectory_error_step,
                    dim=NUM_PARTICLES,
                    inputs=[q_states[t], target_array, terms, t],
                    device=device,
                )
            wp.launch(
                reduce_trajectory_loss,
                dim=1,
                inputs=[terms, loss, NUM_STEPS + 1],
                device=device,
            )

    trajectory = np.stack([q.numpy() for q in q_states])
    reactions = grasp_reactions.numpy().reshape(NUM_STEPS, 2, 3)
    return trajectory, controls, tape, loss, reactions


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def make_ground_truth(device):
    steps = knot_steps()
    controls = np.zeros((len(steps), 2, 3), dtype=np.float32)
    for k, step in enumerate(steps):
        ramp = min(1.0, step / (0.45 * NUM_STEPS))
        controls[k, :, 2] = 1.0 * ramp
    target, _, _, _, reactions = simulate(controls, device, guide_grasps=True)
    grasp_forces = reactions.copy()
    for step in range(NUM_STEPS):
        knot = step // CONTROL_STRIDE
        alpha = (step - knot * CONTROL_STRIDE) / CONTROL_STRIDE
        grasp_forces[step, :, 2] = (
            controls[knot, :, 2] * (1.0 - alpha)
            + controls[knot + 1, :, 2] * alpha
        )
    force_knots = np.empty_like(controls)
    for k, step in enumerate(steps):
        sample = min(step, NUM_STEPS - 1)
        force_knots[k, :, :2] = reactions[sample, :, :2]
        force_knots[k, :, 2] = controls[k, :, 2]
    return target, force_knots, reactions, grasp_forces


def adam_fit(target, initial_controls, device):
    controls = initial_controls.copy()
    first = np.zeros_like(controls)
    second = np.zeros_like(controls)
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8

    for iteration in range(1, ADAM_ITERATIONS + 1):
        _, control_array, tape, loss, _ = simulate(
            controls, device, target=target, differentiate=True, guide_grasps=False
        )
        tape.backward(loss=loss)
        gradient = tape.gradients[control_array].numpy()
        value = float(loss.numpy()[0])

        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        first_hat = first / (1.0 - beta1**iteration)
        second_hat = second / (1.0 - beta2**iteration)
        controls -= ADAM_LEARNING_RATE * first_hat / (np.sqrt(second_hat) + epsilon)
        controls[:, :, 2] = np.maximum(controls[:, :, 2], 0.0)
        print(f"Adam {iteration:04d}/{ADAM_ITERATIONS}: trajectory loss={value:.8e}")
    return controls


def save_results(target, initial, optimized, target_controls, initial_controls, optimized_controls):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    times = np.arange(NUM_STEPS + 1) * DT
    control_times = np.asarray(knot_steps()) * DT

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    colors = {0: "tab:blue", COLS - 1: "tab:orange"}
    for particle in GRASP_PARTICLES:
        axes[0].plot(times, target[:, particle, 2], "--", color=colors[particle],
                     label=f"target p{particle}")
        axes[0].plot(times, optimized[:, particle, 2], color=colors[particle],
                     label=f"optimized p{particle}")
        axes[0].plot(times, initial[:, particle, 2], ":", color=colors[particle],
                     label=f"initial p{particle}")
    axes[0].set_ylabel("grasp Z (m)")
    axes[0].legend(ncol=3)
    components = ["Fx", "Fy", "Fz"]
    for component, (axis, component_name) in enumerate(zip(axes[1:], components)):
        for point, particle in enumerate(GRASP_PARTICLES):
            color = colors[particle]
            axis.plot(control_times, target_controls[:, point, component], "--",
                      color=color, label=f"target {component_name} p{particle}")
            axis.plot(control_times, initial_controls[:, point, component], ":",
                      color=color, alpha=0.65, label=f"initial {component_name} p{particle}")
            axis.plot(control_times, optimized_controls[:, point, component],
                      color=color, label=f"optimized {component_name} p{particle}")
        axis.set_ylabel(f"{component_name} (N)")
        axis.legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trajectory_and_force_comparison.png", dpi=150)
    plt.close(fig)

    trajectories = [target, initial, optimized]
    names = ["Target", "Initial force", "Optimized force"]
    colors = ["tab:blue", "tab:gray", "tab:orange"]
    fig = plt.figure(figsize=(18, 6))
    axes3d = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    def draw(ax, positions, title, color):
        ax.clear()
        for row in range(ROWS):
            for col in range(COLS):
                i = row * COLS + col
                if col + 1 < COLS:
                    j = i + 1
                    ax.plot([positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            [positions[i, 2], positions[j, 2]], color=color, linewidth=1.0)
                if row + 1 < ROWS:
                    j = i + COLS
                    ax.plot([positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            [positions[i, 2], positions[j, 2]], color=color, linewidth=1.0)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color=color, s=18)
        grasp = np.asarray(GRASP_PARTICLES)
        ax.scatter(positions[grasp, 0], positions[grasp, 1], positions[grasp, 2],
                   color="red", s=42)
        ax.set_title(title)
        ax.set_xlim(-0.15, (COLS - 1) * SPACING + 0.15)
        ax.set_ylim(-0.15, (ROWS - 1) * SPACING + 0.15)
        ax.set_zlim(-0.1, 1.0)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=25, azim=-60)

    def update(frame):
        for ax, trajectory, name, color in zip(axes3d, trajectories, names, colors):
            draw(ax, trajectory[frame], name, color)
        fig.suptitle(f"Vertical grasp lift | t={times[frame]:.3f} s")
        return axes3d

    animation = FuncAnimation(fig, update, frames=range(0, NUM_STEPS + 1, 2), interval=50)
    animation.save(OUTPUT_DIR / "vertical_grasp_force_comparison.gif", writer="pillow", fps=20)
    plt.close(fig)


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    print(f"Using device: {device}")
    target, target_controls, target_reactions, target_grasp_forces = make_ground_truth(device)
    initial_controls = np.zeros((len(knot_steps()), 2, 3), dtype=np.float32)
    initial, _, _, _, _ = simulate(initial_controls, device, guide_grasps=False)

    optimized_controls = adam_fit(target, initial_controls, device)
    optimized, _, _, _, _ = simulate(optimized_controls, device, guide_grasps=False)
    save_results(target, initial, optimized, target_controls,
                 initial_controls, optimized_controls)

    np.save(OUTPUT_DIR / "target_trajectory.npy", target)
    np.save(OUTPUT_DIR / "target_force_knots.npy", target_controls)
    np.save(OUTPUT_DIR / "target_grasp_reaction_forces.npy", target_reactions)
    np.save(OUTPUT_DIR / "target_grasp_forces_xyz.npy", target_grasp_forces)
    np.save(OUTPUT_DIR / "initial_force_knots.npy", initial_controls)
    np.save(OUTPUT_DIR / "optimized_force_knots.npy", optimized_controls)
    print(f"Initial trajectory loss: {np.mean((initial - target) ** 2):.8e}")
    print(f"Optimized trajectory loss: {np.mean((optimized - target) ** 2):.8e}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

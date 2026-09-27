"""Tutorial 03: differentiable inverse cloth dynamics with Warp and Adam.

Known grasp particles receive time-varying 3D forces. Warp autodiff computes
the trajectory-loss gradient through the spring-mass simulation; Adam updates
the force controls. The objective contains trajectory loss only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import warp as wp


ROWS = COLS = 6
NUM_PARTICLES = ROWS * COLS
FORCE_PARTICLES = [0, COLS - 1]
SPACING = 0.20
# Approximate a light 1 m x 1 m fabric patch (about 0.18 kg total).
MASS = 0.005
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
ADAM_LEARNING_RATE = 0.5
OUTPUT_DIR = Path("outputs") / "tutorial_03"


@wp.kernel
def spring_mass_step(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    force_knots: wp.array2d(dtype=wp.vec3),
    q_next: wp.array(dtype=wp.vec3),
    qd_next: wp.array(dtype=wp.vec3),
    step: int,
    num_knots: int,
    control_stride: int,
    dt: float,
):
    i = wp.tid()
    row = i // COLS
    col = i - row * COLS
    position = q[i]
    velocity = qd[i]
    force = wp.vec3(0.0, 0.0, -MASS * GRAVITY)

    # Structural and diagonal shear springs connect each particle to its
    # eight immediate grid neighbors. Each particle sums its own forces.
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

    # Piecewise-linear controls, interpolated between force knots.
    knot = step // control_stride
    alpha = float(step - knot * control_stride) / float(control_stride)
    if knot >= num_knots - 1:
        knot = num_knots - 1
        next_knot = knot
        alpha = 0.0
    else:
        next_knot = knot + 1
    external = force_knots[knot, i] * (1.0 - alpha) + force_knots[next_knot, i] * alpha
    force = force + external

    # Differentiable penalty contact with the floor.
    floor_height = GROUND_Z + PARTICLE_RADIUS
    if position[2] < floor_height:
        penetration = floor_height - position[2]
        normal_speed = wp.min(velocity[2], 0.0)
        force = force + wp.vec3(0.0, 0.0, GROUND_STIFFNESS * penetration - GROUND_DAMPING * normal_speed)

    acceleration = force / MASS
    velocity_next = velocity + dt * acceleration
    position_next = position + dt * velocity_next
    qd_next[i] = velocity_next
    q_next[i] = position_next


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


def control_knot_steps():
    steps = list(range(0, NUM_STEPS + 1, CONTROL_STRIDE))
    if steps[-1] != NUM_STEPS:
        steps.append(NUM_STEPS)
    return steps


def initial_positions():
    return np.asarray(
        [(col * SPACING, row * SPACING, GROUND_Z + PARTICLE_RADIUS)
         for row in range(ROWS) for col in range(COLS)],
        dtype=np.float32,
    )


def run_simulation(force_knots_np, device, target_np=None, differentiate=False):
    force_knots = wp.array(
        force_knots_np,
        dtype=wp.vec3,
        device=device,
        requires_grad=differentiate,
    )
    q0 = wp.array(initial_positions(), dtype=wp.vec3, device=device, requires_grad=differentiate)
    qd0 = wp.zeros(NUM_PARTICLES, dtype=wp.vec3, device=device, requires_grad=differentiate)
    q_states = [q0]
    qd_states = [qd0]

    for _ in range(NUM_STEPS):
        q_states.append(wp.empty(NUM_PARTICLES, dtype=wp.vec3, device=device, requires_grad=differentiate))
        qd_states.append(wp.empty(NUM_PARTICLES, dtype=wp.vec3, device=device, requires_grad=differentiate))

    if target_np is not None:
        target = wp.array(target_np, dtype=wp.vec3, device=device)
        terms = wp.empty((NUM_STEPS + 1, NUM_PARTICLES), dtype=float, device=device, requires_grad=differentiate)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=differentiate)
    else:
        target = terms = loss = None

    tape = wp.Tape() if differentiate else None
    context = tape if differentiate else _NullTape()
    with context:
        for step in range(NUM_STEPS):
            wp.launch(
                spring_mass_step,
                dim=NUM_PARTICLES,
                inputs=[q_states[step], qd_states[step], force_knots,
                        q_states[step + 1], qd_states[step + 1], step,
                        len(force_knots_np), CONTROL_STRIDE, DT],
                device=device,
            )
        if target is not None:
            for t in range(NUM_STEPS + 1):
                wp.launch(
                    trajectory_error_step,
                    dim=NUM_PARTICLES,
                    inputs=[q_states[t], target, terms, t],
                    device=device,
                )
            wp.launch(
                reduce_trajectory_loss,
                dim=1,
                inputs=[terms, loss, NUM_STEPS + 1],
                device=device,
            )

    trajectory = np.stack([q.numpy() for q in q_states])
    return trajectory, force_knots, tape, loss


class _NullTape:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def make_target(device):
    steps = control_knot_steps()
    truth = np.zeros((len(steps), NUM_PARTICLES, 3), dtype=np.float32)
    # Synthetic lift generated by upward forces at two known grasp points.
    for knot, step in enumerate(steps):
        ramp = min(1.0, step / (0.45 * NUM_STEPS))
        truth[knot, FORCE_PARTICLES, 2] = 1.0 * ramp
    target, _, _, _ = run_simulation(truth, device)
    return target, truth


def save_outputs(target, initial, prediction, forces):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    time = np.arange(NUM_STEPS + 1) * DT
    control_times = np.asarray(control_knot_steps()) * DT

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for particle in FORCE_PARTICLES:
        axes[0].plot(time, target[:, particle, 2], "--", label=f"target p{particle}")
        axes[0].plot(time, initial[:, particle, 2], ":", label=f"initial p{particle}")
        axes[0].plot(time, prediction[:, particle, 2], label=f"fit p{particle}")
    axes[0].set_ylabel("z position (m)")
    axes[0].legend()
    for particle in FORCE_PARTICLES:
        axes[1].plot(control_times, forces[:, particle, 2], label=f"Fz p{particle}")
    axes[1].set_ylabel("external force (N)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trajectory_and_force.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(18, 6))
    ax_target = fig.add_subplot(1, 3, 1, projection="3d")
    ax_initial = fig.add_subplot(1, 3, 2, projection="3d")
    ax_fit = fig.add_subplot(1, 3, 3, projection="3d")

    def draw(ax, positions, title, color):
        ax.clear()
        for row in range(ROWS):
            for col in range(COLS):
                i = row * COLS + col
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = row + dr, col + dc
                    if nr < ROWS and nc < COLS:
                        j = nr * COLS + nc
                        ax.plot(*zip(positions[i], positions[j]), color=color, linewidth=1.2)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color=color, s=20)
        ax.scatter(positions[FORCE_PARTICLES, 0], positions[FORCE_PARTICLES, 1],
                   positions[FORCE_PARTICLES, 2], color="red", s=45)
        ax.set_title(title)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, (ROWS - 1) * SPACING + 0.2)
        ax.set_zlim(GROUND_Z - 0.1, 0.8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=24, azim=-60)

    def update(frame):
        draw(ax_target, target[frame], "Target trajectory", "tab:blue")
        draw(ax_initial, initial[frame], "Initial force (0 N)", "tab:gray")
        draw(ax_fit, prediction[frame], "Adam fit", "tab:orange")
        fig.suptitle(f"Differentiable cloth force inference | t={time[frame]:.3f} s")
        return ax_target, ax_initial, ax_fit

    animation = FuncAnimation(fig, update, frames=range(0, NUM_STEPS + 1, 2), interval=50)
    animation.save(OUTPUT_DIR / "cloth_trajectory_comparison.gif", writer="pillow", fps=20)
    plt.close(fig)


def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    print(f"Using device: {device}")
    target, ground_truth_forces = make_target(device)
    # Optimize forces only at the known grasp particles; other particle forces
    # stay zero. The simulator still evaluates all 36 particle trajectories.
    reduced = np.zeros((len(control_knot_steps()), len(FORCE_PARTICLES), 3), dtype=np.float32)
    full = np.zeros((len(control_knot_steps()), NUM_PARTICLES, 3), dtype=np.float32)

    # Adam's differentiable parameter array contains only grasp controls.
    # Expand inside the objective by keeping a compact-to-full index mapping.
    # The wrapper also gathers the corresponding gradients back to Adam.
    fitted = optimize_known_grasp_forces(target, reduced, device)
    full[:, FORCE_PARTICLES, :] = fitted
    initial = np.zeros_like(full)
    initial_prediction, _, _, _ = run_simulation(initial, device)
    prediction, _, _, _ = run_simulation(full, device)
    save_outputs(target, initial_prediction, prediction, full)
    np.save(OUTPUT_DIR / "target_trajectory.npy", target)
    np.save(OUTPUT_DIR / "ground_truth_force_knots.npy", ground_truth_forces)
    np.save(OUTPUT_DIR / "initial_force_knots.npy", initial)
    np.save(OUTPUT_DIR / "optimized_force_knots.npy", full)
    print(f"Final trajectory loss: {np.mean((prediction - target) ** 2):.8e}")
    print(f"Saved results to: {OUTPUT_DIR}")


def optimize_known_grasp_forces(target, reduced_controls, device):
    controls = reduced_controls.copy()
    m = np.zeros_like(controls)
    v = np.zeros_like(controls)
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
    for iteration in range(1, ADAM_ITERATIONS + 1):
        full = np.zeros((len(control_knot_steps()), NUM_PARTICLES, 3), dtype=np.float32)
        full[:, FORCE_PARTICLES, :] = controls
        _, force_array, tape, loss = run_simulation(
            full, device, target_np=target, differentiate=True
        )
        tape.backward(loss=loss)
        full_gradient = tape.gradients[force_array].numpy()
        gradient = full_gradient[:, FORCE_PARTICLES, :]
        value = float(loss.numpy()[0])
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * gradient * gradient
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        controls -= ADAM_LEARNING_RATE * m_hat / (np.sqrt(v_hat) + epsilon)
        print(f"Adam {iteration:04d}/{ADAM_ITERATIONS}: trajectory loss={value:.8e}")
    return controls


if __name__ == "__main__":
    main()

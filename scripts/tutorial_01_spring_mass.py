import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import warp as wp


# ============================================================
# Warp initialization
# ============================================================

wp.init()

DEVICE = "cuda" if wp.is_cuda_available() else "cpu"

print(f"Using device: {DEVICE}")


# ============================================================
# Simulation configuration
# ============================================================

ROWS = 6
COLS = 6
N = ROWS * COLS

SPACING = 0.20

NUM_STEPS = 180
DT = 0.002

MASS = 1.0
DAMPING = 5.0

# External downward force
PUSH_FORCE = 18.0
PUSH_STEPS = 70

# Hidden ground-truth stiffness
TRUE_K = 700.0

# Initial guess
INITIAL_K = 120.0

NUM_ITERS = 80
LEARNING_RATE = 0.06


# ============================================================
# Build 2D mass-spring object
# ============================================================

def particle_id(row, col):
    return row * COLS + col


initial_positions = []

for r in range(ROWS):
    for c in range(COLS):

        x = c * SPACING
        y = -r * SPACING

        initial_positions.append([x, y])


initial_positions = np.asarray(
    initial_positions,
    dtype=np.float32,
)


# ============================================================
# Build springs
# ============================================================

spring_i = []
spring_j = []
rest_lengths = []


def add_spring(i, j):
    spring_i.append(i)
    spring_j.append(j)

    p1 = initial_positions[i]
    p2 = initial_positions[j]

    rest_lengths.append(
        np.linalg.norm(p2 - p1)
    )


# Structural springs
for r in range(ROWS):
    for c in range(COLS):

        current = particle_id(r, c)

        # Horizontal spring
        if c < COLS - 1:
            add_spring(
                current,
                particle_id(r, c + 1),
            )

        # Vertical spring
        if r < ROWS - 1:
            add_spring(
                current,
                particle_id(r + 1, c),
            )


# Shear springs
for r in range(ROWS - 1):
    for c in range(COLS - 1):

        add_spring(
            particle_id(r, c),
            particle_id(r + 1, c + 1),
        )

        add_spring(
            particle_id(r, c + 1),
            particle_id(r + 1, c),
        )


spring_i = np.asarray(
    spring_i,
    dtype=np.int32,
)

spring_j = np.asarray(
    spring_j,
    dtype=np.int32,
)

rest_lengths = np.asarray(
    rest_lengths,
    dtype=np.float32,
)

NUM_SPRINGS = len(spring_i)

print(f"Particles: {N}")
print(f"Springs: {NUM_SPRINGS}")


# ============================================================
# Warp arrays
# ============================================================

initial_positions_wp = wp.array(
    initial_positions,
    dtype=wp.vec2,
    device=DEVICE,
)

spring_i_wp = wp.array(
    spring_i,
    dtype=wp.int32,
    device=DEVICE,
)

spring_j_wp = wp.array(
    spring_j,
    dtype=wp.int32,
    device=DEVICE,
)

rest_lengths_wp = wp.array(
    rest_lengths,
    dtype=wp.float32,
    device=DEVICE,
)


# ============================================================
# Physics kernel
# ============================================================

@wp.kernel
def simulate_step(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),

    initial_x: wp.array(dtype=wp.vec2),

    spring_i: wp.array(dtype=wp.int32),
    spring_j: wp.array(dtype=wp.int32),
    rest_length: wp.array(dtype=wp.float32),

    log_k: wp.array(dtype=wp.float32),

    x_next: wp.array(dtype=wp.vec2),
    v_next: wp.array(dtype=wp.vec2),

    num_springs: int,

    rows: int,
    cols: int,

    step: int,
    push_steps: int,

    dt: float,
    mass: float,
    damping: float,
    push_force: float,
):

    particle = wp.tid()

    # --------------------------------------------------------
    # Fix the top row
    # --------------------------------------------------------

    if particle < cols:

        x_next[particle] = initial_x[particle]
        v_next[particle] = wp.vec2(0.0, 0.0)

        return

    # --------------------------------------------------------
    # Recover positive stiffness
    # --------------------------------------------------------

    k = wp.exp(log_k[0])

    total_force = wp.vec2(0.0, 0.0)

    # --------------------------------------------------------
    # Compute spring forces
    # --------------------------------------------------------

    for s in range(num_springs):

        i = spring_i[s]
        j = spring_j[s]

        if particle == i or particle == j:

            other = j

            if particle == j:
                other = i

            delta = x[other] - x[particle]

            distance = wp.length(delta)

            if distance > 0.000001:

                direction = delta / distance

                extension = (
                    distance
                    - rest_length[s]
                )

                spring_force = (
                    k
                    * extension
                    * direction
                )

                total_force = (
                    total_force
                    + spring_force
                )

    # --------------------------------------------------------
    # Velocity damping
    # --------------------------------------------------------

    total_force = (
        total_force
        - damping * v[particle]
    )

    # --------------------------------------------------------
    # Apply a temporary downward force
    #
    # Force is applied near the center of the bottom row.
    # --------------------------------------------------------

    bottom_start = cols * (rows - 1)

    center_left = (
        bottom_start
        + cols // 2
        - 1
    )

    center_right = center_left + 1

    if step < push_steps:

        if (
            particle == center_left
            or particle == center_right
        ):

            total_force = (
                total_force
                + wp.vec2(
                    0.0,
                    -push_force,
                )
            )

    # --------------------------------------------------------
    # Integrate
    # --------------------------------------------------------

    acceleration = total_force / mass

    new_v = (
        v[particle]
        + acceleration * dt
    )

    new_x = (
        x[particle]
        + new_v * dt
    )

    v_next[particle] = new_v
    x_next[particle] = new_x


# ============================================================
# Record trajectory
# ============================================================

@wp.kernel
def record_positions(
    x: wp.array(dtype=wp.vec2),
    trajectory: wp.array(dtype=wp.vec2),
    step: int,
    num_particles: int,
):

    i = wp.tid()

    trajectory[
        step * num_particles + i
    ] = x[i]


# ============================================================
# Trajectory loss
# ============================================================

@wp.kernel
def compute_loss(
    predicted: wp.array(dtype=wp.vec2),
    target: wp.array(dtype=wp.vec2),
    loss: wp.array(dtype=wp.float32),
    num_values: int,
):

    i = wp.tid()

    diff = predicted[i] - target[i]

    error = wp.dot(
        diff,
        diff,
    )

    wp.atomic_add(
        loss,
        0,
        error / float(num_values),
    )


# ============================================================
# Simulator
# ============================================================

def simulate(log_k, requires_grad=False):

    x = wp.array(
        initial_positions,
        dtype=wp.vec2,
        device=DEVICE,
        requires_grad=requires_grad,
    )

    v = wp.zeros(
        N,
        dtype=wp.vec2,
        device=DEVICE,
        requires_grad=requires_grad,
    )

    trajectory = wp.zeros(
        (NUM_STEPS + 1) * N,
        dtype=wp.vec2,
        device=DEVICE,
        requires_grad=requires_grad,
    )

    # Keep every state alive for autodiff
    states = [(x, v)]

    wp.launch(
        record_positions,
        dim=N,
        inputs=[
            x,
            trajectory,
            0,
            N,
        ],
        device=DEVICE,
    )

    for step in range(NUM_STEPS):

        x_next = wp.zeros(
            N,
            dtype=wp.vec2,
            device=DEVICE,
            requires_grad=requires_grad,
        )

        v_next = wp.zeros(
            N,
            dtype=wp.vec2,
            device=DEVICE,
            requires_grad=requires_grad,
        )

        wp.launch(
            simulate_step,
            dim=N,
            inputs=[
                x,
                v,

                initial_positions_wp,

                spring_i_wp,
                spring_j_wp,
                rest_lengths_wp,

                log_k,

                x_next,
                v_next,

                NUM_SPRINGS,

                ROWS,
                COLS,

                step,
                PUSH_STEPS,

                DT,
                MASS,
                DAMPING,
                PUSH_FORCE,
            ],
            device=DEVICE,
        )

        x = x_next
        v = v_next

        states.append((x, v))

        wp.launch(
            record_positions,
            dim=N,
            inputs=[
                x,
                trajectory,
                step + 1,
                N,
            ],
            device=DEVICE,
        )

    return trajectory, states


# ============================================================

def trajectory_to_numpy(
    trajectory,
):

    array = trajectory.numpy()

    return array.reshape(
        NUM_STEPS + 1,
        N,
        2,
    )


# ============================================================
# Visualization helper
# ============================================================

def draw_object(
    ax,
    positions,
    title,
):

    ax.clear()

    # --------------------------------------------------------
    # Draw springs
    # --------------------------------------------------------

    for i, j in zip(
        spring_i,
        spring_j,
    ):

        p1 = positions[i]
        p2 = positions[j]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linewidth=1.0,
        )

    # --------------------------------------------------------
    # Draw particles
    # --------------------------------------------------------

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=30,
    )

    # Highlight fixed particles
    ax.scatter(
        positions[:COLS, 0],
        positions[:COLS, 1],
        s=60,
        marker="s",
    )

    # Highlight pushed particles
    bottom_start = COLS * (ROWS - 1)

    pushed = [
        bottom_start + COLS // 2 - 1,
        bottom_start + COLS // 2,
    ]

    ax.scatter(
        positions[pushed, 0],
        positions[pushed, 1],
        s=80,
        marker="o",
    )

    ax.set_title(title)

    ax.set_xlim(
        -0.15,
        (COLS - 1) * SPACING + 0.15,
    )

    ax.set_ylim(
        -(ROWS - 1) * SPACING - 0.65,
        0.15,
    )

    ax.set_aspect(
        "equal",
    )

    ax.grid(
        alpha=0.2,
    )


# ============================================================

def main():
    output_dir = Path("outputs") / "tutorial_01"
    output_dir.mkdir(exist_ok=True)

    # Ground-truth observation
    # ============================================================
    
    true_log_k = wp.array(
        [math.log(TRUE_K)],
        dtype=wp.float32,
        device=DEVICE,
    )
    
    observed_trajectory, _ = simulate(
        true_log_k,
        requires_grad=False,
    )
    
    
    # ============================================================
    # Initial simulation for comparison
    # ============================================================
    
    initial_log_k = wp.array(
        [math.log(INITIAL_K)],
        dtype=wp.float32,
        device=DEVICE,
    )
    
    initial_trajectory, _ = simulate(
        initial_log_k,
        requires_grad=False,
    )

    ground_truth_np = trajectory_to_numpy(
        observed_trajectory
    )

    initial_np = trajectory_to_numpy(
        initial_trajectory
    )
    
    
    # ============================================================
    # Inverse modeling
    # ============================================================
    
    log_k_value = math.log(INITIAL_K)
    
    k_history = []
    loss_history = []
    
    
    # Adam states
    adam_m = 0.0
    adam_v = 0.0
    
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    
    print()
    print("Starting inverse optimization")
    print("-" * 50)
    
    
    for iteration in range(1, NUM_ITERS + 1):
    
        log_k = wp.array(
            [log_k_value],
            dtype=wp.float32,
            device=DEVICE,
            requires_grad=True,
        )
    
        loss = wp.zeros(
            1,
            dtype=wp.float32,
            device=DEVICE,
            requires_grad=True,
        )
    
        tape = wp.Tape()
    
        with tape:
    
            predicted_trajectory, states = simulate(
                log_k,
                requires_grad=True,
            )
    
            num_values = (
                (NUM_STEPS + 1)
                * N
            )
    
            wp.launch(
                compute_loss,
                dim=num_values,
                inputs=[
                    predicted_trajectory,
                    observed_trajectory,
                    loss,
                    num_values,
                ],
                device=DEVICE,
            )
    
        tape.backward(loss)
    
        gradient = float(
            log_k.grad.numpy()[0]
        )
    
        current_loss = float(
            loss.numpy()[0]
        )
    
        # --------------------------------------------------------
        # Adam update
        # --------------------------------------------------------
    
        adam_m = (
            beta1 * adam_m
            + (1.0 - beta1) * gradient
        )
    
        adam_v = (
            beta2 * adam_v
            + (1.0 - beta2)
            * gradient * gradient
        )
    
        m_hat = (
            adam_m
            / (1.0 - beta1 ** iteration)
        )
    
        v_hat = (
            adam_v
            / (1.0 - beta2 ** iteration)
        )
    
        log_k_value -= (
            LEARNING_RATE
            * m_hat
            / (
                math.sqrt(v_hat)
                + epsilon
            )
        )
    
        current_k = math.exp(
            log_k_value
        )
    
        k_history.append(
            current_k
        )
    
        loss_history.append(
            current_loss
        )
    
        if (
            iteration == 1
            or iteration % 10 == 0
        ):
    
            print(
                f"Iteration {iteration:03d} | "
                f"k = {current_k:8.2f} | "
                f"loss = {current_loss:.8f}"
            )
    
    
    estimated_k = math.exp(
        log_k_value
    )
    
    
    print()
    print("=" * 50)
    print(f"Ground truth k : {TRUE_K:.2f}")
    print(f"Initial k      : {INITIAL_K:.2f}")
    print(f"Estimated k    : {estimated_k:.2f}")
    print("=" * 50)
    
    
    # ============================================================
    # Final simulation
    # ============================================================
    
    estimated_log_k = wp.array(
        [math.log(estimated_k)],
        dtype=wp.float32,
        device=DEVICE,
    )
    
    estimated_trajectory, _ = simulate(
        estimated_log_k,
        requires_grad=False,
    )

    estimated_np = trajectory_to_numpy(
        estimated_trajectory
    )
    
    
    # ============================================================
    # Convert trajectories to NumPy
    # ============================================================
    
    # Animated comparison
    # ============================================================
    
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )
    
    
    def update(frame):
    
        draw_object(
            axes[0],
            ground_truth_np[frame],
            f"Ground Truth\nk = {TRUE_K:.0f}",
        )
    
        draw_object(
            axes[1],
            initial_np[frame],
            f"Initial Guess\nk = {INITIAL_K:.0f}",
        )
    
        draw_object(
            axes[2],
            estimated_np[frame],
            f"Optimized\nk = {estimated_k:.1f}",
        )
    
        time = frame * DT
    
        fig.suptitle(
            f"Inverse Modeling of Spring Stiffness | "
            f"t = {time:.3f} s",
            fontsize=16,
            y=0.98,
        )
    
        return axes
    
    
    animation = FuncAnimation(
        fig,
        update,
        frames=range(
            0,
            NUM_STEPS + 1,
            2,
        ),
        interval=40,
        repeat=True,
    )
    
    fig.subplots_adjust(
        top=0.82,
        bottom=0.08,
        left=0.03,
        right=0.97,
        wspace=0.12,
    )
    
    animation.save(output_dir / "spring_mass_comparison.gif", writer="pillow", fps=25)
    plt.close(fig)
    
    
    # ============================================================
    # Optimization visualization
    # ============================================================
    
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )
    
    
    # ------------------------------------------------------------
    # Estimated stiffness
    # ------------------------------------------------------------
    
    axes[0].plot(
        k_history,
        linewidth=2,
    )
    
    axes[0].axhline(
        TRUE_K,
        linestyle="--",
        label=f"Ground Truth = {TRUE_K}",
    )
    
    axes[0].set_xlabel(
        "Optimization iteration"
    )
    
    axes[0].set_ylabel(
        "Spring stiffness k"
    )
    
    axes[0].set_title(
        "Estimated Physical Parameter"
    )
    
    axes[0].legend()
    
    axes[0].grid(
        alpha=0.3
    )
    
    
    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------
    
    axes[1].plot(
        loss_history,
        linewidth=2,
    )
    
    axes[1].set_yscale(
        "log"
    )
    
    axes[1].set_xlabel(
        "Optimization iteration"
    )
    
    axes[1].set_ylabel(
        "Trajectory loss"
    )
    
    axes[1].set_title(
        "Inverse Modeling Loss"
    )
    
    axes[1].grid(
        alpha=0.3
    )
    
    
    plt.tight_layout()
    
    plt.savefig(output_dir / "optimization_history.png", dpi=150)
    plt.close(fig)
    
    
    # ============================================================
    # End-point trajectory visualization
    # ============================================================
    
    bottom_center = (
        COLS * (ROWS - 1)
        + COLS // 2
    )
    
    time = (
        np.arange(NUM_STEPS + 1)
        * DT
    )
    
    
    plt.figure(
        figsize=(9, 5)
    )
    
    
    plt.plot(
        time,
        ground_truth_np[
            :,
            bottom_center,
            1,
        ],
        label="Ground Truth",
        linewidth=3,
    )
    
    
    plt.plot(
        time,
        initial_np[
            :,
            bottom_center,
            1,
        ],
        label="Initial Guess",
        linestyle="--",
    )
    
    
    plt.plot(
        time,
        estimated_np[
            :,
            bottom_center,
            1,
        ],
        label="Optimized",
        linestyle=":",
        linewidth=3,
    )
    
    
    plt.xlabel(
        "Time [s]"
    )
    
    plt.ylabel(
        "Vertical position"
    )
    
    plt.title(
        "Observed vs Simulated Motion"
    )
    
    plt.legend()
    
    plt.grid(
        alpha=0.3
    )
    
    plt.tight_layout()
    
    plt.savefig(output_dir / "trajectory_comparison.png", dpi=150)
    plt.close()
    
    
    # ============================================================
if __name__ == "__main__":
    main()



import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(episode_rewards, window=100, smooth=50):
    rewards = np.array(episode_rewards)

    # --- convergence ---
    convergence = []
    for i in range(2 * window, len(rewards)):
        last = np.mean(rewards[i-window:i]) # last avg reward
        prev = np.mean(rewards[i-2*window:i-window]) # previous avg reward
        convergence.append(last - prev) # append the difference (> 0 --> improvement; < 0 --> worsening; = 0 --> plateau)

    convergence = np.array(convergence)

    # --- smoothing  ---
    if len(convergence) >= smooth:
        convergence_smooth = np.convolve(
            convergence,
            np.ones(smooth)/smooth,
            mode='valid'
        )
    else:
        convergence_smooth = convergence

    # --- plot ---
    plt.figure(figsize=(10, 5))

    plt.plot(convergence, alpha=0.3, label="Raw Δ")
    plt.plot(
        range(len(convergence_smooth)),
        convergence_smooth,
        linewidth=2,
        label=f"Smoothed Δ ({smooth})"
    )

    plt.axhline(0, linestyle='--')

    plt.title("Convergence (Δ Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Δ Reward")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
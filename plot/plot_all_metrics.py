import numpy as np
import matplotlib.pyplot as plt


def plot_training_metrics(
    episode_rewards,
    episode_epsilons=None,
    window=100,
    title="Training Metrics",
    save_path=None
):
    """
    Plot:
    - reward per episode
    - moving average
    - convergence (delta tra finestre)
    - epsilon (opzionale)

    Args:
        episode_rewards (list)
        episode_epsilons (list or None)
        window (int): dimensione finestra per moving avg e convergence
        title (str)
        save_path (str or None)
    """

    rewards = np.array(episode_rewards)

    # --- Moving average ---
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        moving_avg = np.array([])

    # --- Convergence (delta tra finestre) ---
    convergence = []
    for i in range(2 * window, len(rewards)):
        last = np.mean(rewards[i-window:i])
        prev = np.mean(rewards[i-2*window:i-window])
        convergence.append(last - prev)

    convergence = np.array(convergence)

    for i, delta in enumerate(convergence):
        if abs(delta) < 0.01:
            print(f"[INFO] Convergence reached around episode {i + 2*window}")
            break

    # --- Plot ---
    fig, axs = plt.subplots(3 if episode_epsilons is not None else 2, 1, figsize=(10, 10))

    # ===== REWARD =====
    axs[0].plot(rewards, alpha=0.3, label="Reward")
    if len(moving_avg) > 0:
        axs[0].plot(range(window-1, len(rewards)), moving_avg, label=f"Moving Avg ({window})", linewidth=2)
    axs[0].set_title("Episode Reward")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid()

    # ===== CONVERGENCE =====
    axs[1].plot(range(2*window, len(rewards)), convergence, label="Convergence Δ")
    axs[1].axhline(0, linestyle='--')
    axs[1].set_title("Convergence (Delta between windows)")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Δ Reward")
    axs[1].legend()
    axs[1].grid()

    # ===== EPSILON =====
    if episode_epsilons is not None:
        axs[2].plot(episode_epsilons, label="Epsilon", color="orange")
        axs[2].set_title("Epsilon Decay")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Epsilon")
        axs[2].legend()
        axs[2].grid()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
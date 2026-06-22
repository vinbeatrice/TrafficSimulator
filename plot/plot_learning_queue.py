import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(episode_rewards):
    rewards = np.array(episode_rewards)

    window = 50
    moving_avg = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Reward per episode")
    plt.plot(
        range(window - 1, len(rewards)),
        moving_avg,
        label=f"Moving average+ ({window})",
        linewidth=2
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=100):

    if len(values) < window:
        return np.array(values)

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


def plot_learning_curve_with_baseline(episode_rewards, baseline_rewards, save_path=None, window=100):

    plt.figure(figsize=(12, 6))

    # agent
    agent_ma = moving_average(
        episode_rewards,
        window
    )

    # baseline
    baseline_ma = moving_average(
        baseline_rewards,
        window
    )

    x_agent = np.arange(
        len(agent_ma)
    ) + window - 1

    x_base = np.arange(
        len(baseline_ma)
    ) + window - 1

    plt.plot(
        x_agent,
        agent_ma,
        label="Agent",
        linewidth=2
    )

    plt.plot(
        x_base,
        baseline_ma,
        label="Baseline",
        linewidth=2
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.title(
        f"Learning Curve (Moving Average {window})"
    )

    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
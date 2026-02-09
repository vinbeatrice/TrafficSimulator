import matplotlib.pyplot as plt
import numpy as np

def plot_learning_queue(episode_rewards):
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

    plt.xlabel("Episoded")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

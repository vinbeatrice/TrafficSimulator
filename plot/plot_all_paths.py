import matplotlib.pyplot as plt
import numpy as np

def plot_all_paths(reward_per_path):
    plt.figure(figsize=(10, 6))

    for path_id, rewards in reward_per_path.items():
        if len(rewards) < 20:
            continue

        smoothed = np.convolve(
            rewards,
            np.ones(50)/50,
            mode="valid"
        )

        plt.plot(smoothed, label=f"Path {path_id}")

    plt.xlabel("Episodes (per path)")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Reward per path (moving average)")
    plt.grid(True)
    plt.show()
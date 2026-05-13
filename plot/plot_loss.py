import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses, smooth=100):
    losses = np.array(losses)

    plt.figure(figsize=(10,5))
    plt.plot(losses, alpha=0.3, label="Raw loss")

    if len(losses) >= smooth:
        smooth_loss = np.convolve(
            losses,
            np.ones(smooth)/smooth,
            mode='valid'
        )
        plt.plot(smooth_loss, linewidth=2, label="Smoothed loss")

    plt.xlabel("Training step")
    plt.ylabel("TD Loss")
    plt.title("DQN Loss")
    plt.legend()
    plt.grid()
    plt.show()
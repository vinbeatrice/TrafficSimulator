import matplotlib.pyplot as plt
import numpy as np

def plot_epsilon(epsilons):
    episodes = np.arange(len(epsilons))

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon decay over episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_gradient_norm(grad_norms):
    grad_norms = np.array(grad_norms)

    plt.figure()
    plt.plot(grad_norms)
    plt.yscale("log")  # FONDAMENTALE
    plt.xlabel("Training step")
    plt.ylabel("Gradient L2 norm (log scale)")
    plt.title("Gradient Norms During Training")
    plt.grid(True)
    plt.show()
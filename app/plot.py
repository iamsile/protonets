import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # enable interactive mode
    plt.ion()
    # initialize figure
    fig, ax = plt.subplots()
    for i in range(100):
        ax.plot(1 + np.random.randn(), 1 + np.random.randn(), "r-o")
        plt.draw()
        plt.pause(0.05)
    # disable interactive mode
    plt.ioff()

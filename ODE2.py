import numpy as np
import matplotlib.pyplot as plt


def concentration(t, k, g):
    C = np.log(g)

    return (np.exp(-k * t + C) - g) / (-k)


def plot_concentration(t_range, k, g):

    x = [concentration(t, k, g) for t in t_range]
    plt.plot(t_range, x, label=f"g = {g}, k = {k}")
    plt.xlabel("t", fontsize=14)
    plt.ylabel("x(t)", fontsize=14)
    plt.grid(visible=True)


stepsize = 0.001
t_range = [t for t in np.arange(0, 5 + stepsize, stepsize)]

plt.figure()
plot_concentration(t_range, g=2, k=3)
plot_concentration(t_range, g=1, k=1.5)
plot_concentration(t_range, g=2, k=2)
plot_concentration(t_range, g=1, k=1)
plt.legend()
plt.savefig("figures/4e.png")

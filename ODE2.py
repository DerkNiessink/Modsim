import numpy as np
import matplotlib.pyplot as plt


def concentration(t, k, g):
    C = np.log(g)
    return (np.exp(-k * t + C) - g) / (-k)


def plot_concentration(t_range, x, k, g):

    plt.plot(t_range, x, label=f"g = {g}, k = {k}")
    plt.xlabel("t", fontsize=14)
    plt.ylabel("x(t)", fontsize=14)
    plt.grid(visible=True)


def func(x, t, g, k):
    return g - k * x


def euler(stepsize, a, b, initial_condition, func, g, k):
    x_estimate_list, t_list = [], []
    x_previous = initial_condition
    t_previous = a
    int_sum = 0
    while a < b + stepsize:
        x_next = x_previous + stepsize * func(x_previous, t_previous, g, k)
        x_estimate_list.append(x_previous)
        t_list.append(a)
        a += stepsize

        int_sum += (x_next + x_previous) / 2 * stepsize

        x_previous = x_next
        t_previous += stepsize

    return t_list, x_estimate_list


stepsize = 0.001
t_range = [t for t in np.arange(0, 5 + stepsize, stepsize)]

params = [(2, 3), (1, 1.5), (2, 2), (1, 1)]


plt.figure()
for g, k in params:
    x = [concentration(t, k, g) for t in t_range]
    plot_concentration(t_range, x, k, g)
plt.legend()
plt.savefig("figures/4e.png")


plt.figure()
for g, k in params:
    t, x = euler(0.001, 0, 5, 0, func, g, k)
    plot_concentration(t, x, k, g)
plt.legend()
plt.savefig("figures/4f.png")

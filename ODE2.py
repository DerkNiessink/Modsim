import numpy as np
import matplotlib.pyplot as plt


def concentration(t, k, g):
    C = np.log(g)
    return (np.exp(-k * t + C) - g) / (-k)


def concentration_time(t, k, g, g_func):
    C = np.log(g_func(t, g, k))
    return (np.exp(-k * t + C) - g_func(t, g, k)) / (-k)


def g_func1(t, g, k):
    return -2 * g * (np.exp(-k * t) - 1)


def g_func2(t, g, k):
    return 2 * g * (np.exp(-k * t) + 1)


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


"""
4e
"""
plt.figure()
for g, k in params:
    x = [concentration(t, k, g) for t in t_range]
    plot_concentration(t_range, x, k, g)
plt.legend()
plt.savefig("figures/4e.png")


"""
4f
"""
plt.figure()
for g, k in params:
    t, x = euler(0.001, 0, 5, 0, func, g, k)
    plot_concentration(t, x, k, g)
plt.legend()
plt.savefig("figures/4f.png")


"""
4g
"""
plt.figure(figsize=(8, 6))
for g, k in params:
    x_list = [concentration(t, k, g) for t in t_range]
    dx_dt_list = [func(x, 0, g, k) for x in x_list]
    plt.plot(x_list, dx_dt_list, label=f"g = {g}, k = {k}")
plt.grid(visible=True)
plt.ylabel(r"$dx/dt$", fontsize=16)
plt.xlabel(r"$x$", fontsize=16)
plt.legend()
plt.savefig("figures/4h.png")


"""
4i
"""
plt.figure()
g1 = [g_func1(t, 1, 1) for t in t_range]
g2 = [g_func2(t, 1, 1) for t in t_range]
plt.plot(t_range, g1, label=r"$g(t) = -2g_0(e^{-kt}-1)$")
plt.plot(t_range, g2, label=r"$g(t) = 2g_0(e^{-kt}+1)$")
plt.xlabel("t", fontsize=16)
plt.ylabel("g(t)", fontsize=16)
plt.grid(visible=True)
plt.legend()
plt.savefig("figures/4i_1.png")

plt.figure()
x_gt = [concentration_time(t, 1, 1, g_func1) for t in t_range]
x_gt2 = [concentration_time(t, 1, 1, g_func2) for t in t_range]
x = [concentration(t, 1, 1) for t in t_range]
x2 = [concentration(t, 1, 2) for t in t_range]
plt.plot(t_range, x_gt, "--", label=r"$g(t) = -2g_0(e^{-kt}-1)$")
plt.plot(t_range, x_gt2, "--", label=r"$g(t) = 2g_0(e^{-kt}+1)$")
plt.plot(t_range, x, label=r"$g(t) = g = g_0$")
plt.plot(t_range, x2, label=r"$g(t) = g = 2g_0$")
plt.xlabel("t", fontsize=16)
plt.ylabel("x(t)", fontsize=16)
plt.grid(visible=True)
plt.legend()
plt.savefig("figures/4i_2.png")

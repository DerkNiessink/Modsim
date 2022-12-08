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


def rabbit_func(x, t, r, k):
    return r * x**2 - k * x


def rabbit_real_func(x, t, r, k):
    return x * (1 - x / r)


def rabbit_die_func(x, t, r, k):
    return x * (1 - x / r - k)


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

"""
5h
"""
plt.figure()
t, x = euler(0.001, 0, 3, 0.5, rabbit_func, 1, 1)
t2, x2 = euler(0.001, 0, 3, 1, rabbit_func, 1, 1)
t3, x3 = euler(0.001, 0, 1, 1.5, rabbit_func, 1, 1)
plt.plot(t, x, label=r"$x_0$ = 0.5")
plt.plot(t2, x2, label=r"$x_0$ = 1")
plt.plot(t3, x3, label=r"$x_0$ = 1.5")
plt.xlabel("t", fontsize=16)
plt.ylabel("x(t)", fontsize=16)
plt.ylim(0, 3)
plt.grid(visible=True)
plt.legend()
plt.savefig("figures/5h.png")

"""
5i
"""
plt.figure()
values = [(3, 0.1), (0.5, 3), (1.5, 1.5)]
for x_max, x_init in values:
    t, x = euler(0.001, 0, 10, x_init, rabbit_real_func, x_max, None)
    plt.plot(t, x, label=f"$x_0 = {x_init}, x_{{max}} = {x_max}$")

plt.xlabel("t", fontsize=16)
plt.ylabel("x(t)", fontsize=16)
plt.grid(visible=True)
plt.legend()
plt.savefig("figures/5i.png")

"""
5j
"""
plt.figure()
x_max = 6
x_list = [x for x in np.arange(-1, x_max + 1, 0.001)]
dx_dt = [rabbit_real_func(x, None, x_max, None) for x in x_list]
plt.plot(x_list, dx_dt)
plt.xlabel("x", fontsize=16)
plt.ylabel(r"$dx/dt$", fontsize=16)
plt.grid(visible=True)
plt.savefig("figures/5l.png")

"""
5o
"""
plt.figure()
x_max = 6
r = 2
stepsizes = [0.25, 0.19, 0.16, 0.1]
for stepsize in stepsizes:
    t, x = euler(stepsize, 0, 3, 25, rabbit_die_func, x_max, r)
    plt.plot(t, x, label=f"step size = {stepsize}")

plt.xlabel(r"$t$", fontsize=16)
plt.ylabel(r"$x(t)$", fontsize=16)
plt.ylim(-2, 10)
plt.legend()
plt.grid(visible=True)
plt.savefig("figures/5o.png")

"""
"ODE.py"

Developed by:
Jenna de Vries and Derk Niessink

At the bottom of this module (indicated by the line) we implemented the
exercises in order. Functions used in these exercises can be quickly accessed
by CTRL+clicking the function call.
"""


import numpy as np
import matplotlib.pyplot as plt


def func_a(x, t):
    return 1


def func_b(x, t):
    return 2 * t


def func_c(x, t):
    return -x


def euler_plot(
    stepsize, a, b, initial_condition, func, label, color, alpha, plot_label
):
    """
    Creates a plot according to the euler approximation given the stepsize, a range
    (a,b), an initial condition and a function. Further is it possible to specify
    the label, color, transparancy and if you want the labels plotted or not.
    """
    x_estimate_list, t_list = [], []
    x_previous = initial_condition
    t_previous = a
    int_sum = 0
    while a < b + stepsize:
        x_next = x_previous + stepsize * func(x_previous, t_previous)
        x_estimate_list.append(x_previous)
        t_list.append(a)
        a += stepsize

        int_sum += (x_next + x_previous) / 2 * stepsize

        x_previous = x_next
        t_previous += stepsize

    print(label, stepsize, int_sum)
    plt.ylim(-5, 5)
    plt.xlim(0, 3)

    if plot_label == True:
        plt.plot(t_list, x_estimate_list, label=label, alpha=alpha, color=color)
    else:
        plt.plot(t_list, x_estimate_list, alpha=alpha, color=color)

    pass


def func1(t):
    return t


def func2(t):
    return (t**2) - 4


def func3(t):
    return 4 * (np.e ** (-t))


def analytical(func, plot_label):
    """
    Plots the exact solution of a given function.
    """
    x_list, t_list = [], []
    for t in np.linspace(0, 3, 100):
        x = func(t)
        t_list.append(t)
        x_list.append(x)

    if plot_label == True:
        plt.plot(t_list, x_list, alpha=0.5, color="black", label="analytical solution")
    else:
        plt.plot(t_list, x_list, alpha=0.5, color="black")


def exercise_3():
    """
    Loops through different stepsizes and saves a plot for exercise 3
    """
    a, b = 0, 3
    stepsize_list = [1, 0.1, 0.01]
    alpha_list = [0.15, 0.4, 0.75]
    for index in range(0, len(stepsize_list)):

        alpha = alpha_list[index]
        plot_label = False
        if index == 2:
            plot_label = True

        euler_plot(
            stepsize_list[index],
            a,
            b,
            0,
            func_a,
            r"$\frac{dx}{dt}=1$, $x(0)=0$",
            "r",
            alpha,
            plot_label,
        )
        euler_plot(
            stepsize_list[index],
            a,
            b,
            -4,
            func_b,
            r"$\frac{dx}{dt}=2t, x(0)=-4$",
            "b",
            alpha,
            plot_label,
        )
        euler_plot(
            stepsize_list[index],
            a,
            b,
            4,
            func_c,
            r"$\frac{dx}{dt}=-x, x(0)=4$",
            "g",
            alpha,
            plot_label,
        )

    analytical(func1, plot_label=False)
    analytical(func2, plot_label=False)
    analytical(func3, plot_label=True)

    plt.xlabel("t", fontsize=16)
    plt.ylabel("x(t)", fontsize=16)
    plt.legend()
    plt.savefig("3.png")

    pass


def runge_kutta_2(
    stepsize, a, b, initial_condition, func, label, color, alpha, plot_label
):
    """
    Creates a plot according to the 2nd order Runge Kutta approximation given the stepsize,
    a range (a,b), an initial condition and a function. Further is it possible to specify
    the label, color, transparancy and if you want the labels plotted or not.
    """
    x_estimate_list, t_list = [], []
    x_previous = initial_condition
    t_previous = a
    int_sum = 0
    while a < b + stepsize:
        k1 = stepsize * func(x_previous, t_previous)
        k2 = stepsize * func(x_previous + 1 / 2, t_previous + (k1 + stepsize) / 2)
        x_next = x_previous + k2
        x_estimate_list.append(x_previous)
        t_list.append(a)
        a += stepsize

        int_sum += (x_next + x_previous) / 2 * stepsize

        x_previous = x_next
        t_previous += stepsize

    print(label, stepsize, int_sum)
    plt.ylim(-5, 5)
    plt.xlim(0, 3)

    if plot_label == True:
        plt.plot(t_list, x_estimate_list, label=label, alpha=alpha, color=color)
    else:
        plt.plot(t_list, x_estimate_list, alpha=alpha, color=color)

    pass


def exercise_3_bonus():
    """
    Loops through different stepsizes and saves a plot for the bonus of exercise 3.
    """
    a, b = 0, 3
    stepsize_list = [1, 0.1, 0.01]
    alpha_list = [0.15, 0.4, 0.75]
    for index in range(0, len(stepsize_list)):

        alpha = alpha_list[index]
        plot_label = False
        if index == 2:
            plot_label = True

        runge_kutta_2(
            stepsize_list[index],
            a,
            b,
            0,
            func_a,
            r"$\frac{dx}{dt}=1$, $x(0)=0$",
            "r",
            alpha,
            plot_label,
        )
        runge_kutta_2(
            stepsize_list[index],
            a,
            b,
            -4,
            func_b,
            r"$\frac{dx}{dt}=2t, x(0)=-4$",
            "b",
            alpha,
            plot_label,
        )
        runge_kutta_2(
            stepsize_list[index],
            a,
            b,
            4,
            func_c,
            r"$\frac{dx}{dt}=-x, x(0)=4$",
            "g",
            alpha,
            plot_label,
        )

    analytical(func1, plot_label=False)
    analytical(func2, plot_label=False)
    analytical(func3, plot_label=True)

    plt.xlabel("t", fontsize=16)
    plt.ylabel("x(t)", fontsize=16)
    plt.legend()
    plt.savefig("3_bonus.png")

    pass


def rabbit_solution(r, k, t, N_i):
    x = N_i * (np.e ** ((r - k) * t))
    return x


def fixed_point(xlist):
    """
    Calculates the fixed point in a list, if there is a difference
    smaller than 0.005 between to iterations of a list it will accept
    the point as a fixed point.
    """
    for i in range(0, len(xlist) - 1):
        if round(xlist[i] - xlist[i + 1], 2) == 0:
            return i


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


def concentration_func(x, t, g, k):
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


"""EXERCISES BEGIN HERE"""
"==================================================================================================="


"""
3
"""
plt.figure()
exercise_3()
plt.clf()


"""
3 bonus
"""
plt.figure()
exercise_3_bonus()
plt.clf()


"""
4e
"""
stepsize = 0.001
t_range = [t for t in np.arange(0, 5 + stepsize, stepsize)]
params = [(2, 3), (1, 1.5), (2, 2), (1, 1)]

plt.figure()
for g, k in params:
    x = [concentration(t, k, g) for t in t_range]
    plot_concentration(t_range, x, k, g)
plt.legend()
plt.savefig("4e.png")


"""
4f
"""
plt.figure()
for g, k in params:
    t, x = euler(0.001, 0, 5, 0, concentration_func, g, k)
    plot_concentration(t, x, k, g)
plt.legend()
plt.savefig("4f.png")


"""
4g
"""
plt.figure(figsize=(8, 6))
for g, k in params:
    x_list = [concentration(t, k, g) for t in t_range]
    dx_dt_list = [concentration_func(x, 0, g, k) for x in x_list]
    plt.plot(x_list, dx_dt_list, label=f"g = {g}, k = {k}")
plt.grid(visible=True)
plt.ylabel(r"$dx/dt$", fontsize=16)
plt.xlabel(r"$x$", fontsize=16)
plt.legend()
plt.savefig("4h.png")


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
plt.savefig("4i_1.png")

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
plt.savefig("4i_2.png")


"""
4k
"""
plt.figure()
values = [(3, 0.1), (0.5, 3), (1.5, 1.5)]

t_list = np.linspace(0, 10)
plt.plot(t_list, t_list, label="t")

x_dict = {}
fixed_point_list = []
for x_max, x_init in values:
    t, x = euler(0.001, 0, 10, x_init, rabbit_real_func, x_max, None)
    plt.plot(t, x, label=f"$x_0 = {x_init}, x_{{max}} = {x_max}$")

    fixed_points = [i for i, j in zip(t, x) if round(i, 2) == round(j, 2)]
    fixed_point_list.append(np.mean(fixed_points))

print(values)
print(fixed_point_list)

plt.scatter(fixed_point_list, fixed_point_list)
plt.xlabel("t", fontsize=16)
plt.ylabel("x(t)", fontsize=16)
plt.grid(visible=True)
plt.legend()
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.savefig("5k.png")


"""
5c/d
"""
plt.figure()
values = [(1, 1), (1, 2), (2, 1)]
N_i = 25
t_list = np.linspace(0, 30, 1000)
for r, k in values:
    x_list = [rabbit_solution(r, k, t, N_i) for t in t_list]
    plt.plot(t_list, x_list, label=f"r = {r}, k = {k}")
plt.grid(visible=True)
plt.xlim(0, 30)
plt.ylim(-5, 200)

plt.xlabel("t", fontsize=16)
plt.ylabel("x(t)", fontsize=16)
plt.legend()
plt.savefig("5cd.png")


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
plt.savefig("5h.png")


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
plt.savefig("5i.png")


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
plt.savefig("5l.png")


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
plt.savefig("5o.png")

# 2
import numpy as np
import matplotlib.pyplot as plt

# a
def func_a(x, t):
    return 1

def func_b(x, t):
    return 2*t

def func_c(x,t):
    return -x

def euler(stepsize, a, b, initial_condition, func, name, label, color, alpha):
    x_estimate_list, t_list = [], []
    x_previous = initial_condition
    t_previous = a
    while a < b+stepsize:
        x_next = x_previous + stepsize * func(x_previous, t_previous)
        x_estimate_list.append(x_next)
        t_list.append(a)
        a += stepsize
        x_previous = x_next
        t_previous += stepsize

    plt.ylim(-5, 5)
    plt.xlim(0,3)
    plt.plot(t_list, x_estimate_list, label=label, alpha=alpha, color=color)
    plt.legend()
    plt.savefig(name)

a, b = 0, 3
stepsize_list=[1,0.1,0.01]
alpha_list=[0.4, 0.6, 0.8]
for index in range(0,len(stepsize_list)):
    alpha = alpha_list[index]
    euler(stepsize_list[index], a, b, 0, func_a, 'a.png', 'a', 'r', alpha)
    euler(stepsize_list[index], a, b, -4, func_b, 'a.png', 'b', 'b', alpha)
    euler(stepsize_list[index], a, b, 4, func_c, 'a.png', 'c', 'g', alpha)
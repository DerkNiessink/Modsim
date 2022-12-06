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

def euler(stepsize, a, b, initial_condition, func, name, label, color):
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
    plt.plot(t_list, x_estimate_list, label=label, alpha=0.4, color=color)
    plt.legend()
    plt.savefig(name)

a, b = 0, 3
stepsize_list=[1,0.1,0.01]
alpha_list=[0.4, 0.6, 0.8]
for stepsize in stepsize_list:
    euler(stepsize, a, b, 0, func_a, 'a.png', 'a', 'r')
    euler(stepsize, a, b, -4, func_b, 'a.png', 'b', 'b')
    euler(stepsize, a, b, 4, func_c, 'a.png', 'c', 'g')
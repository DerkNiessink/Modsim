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

def euler(stepsize, a, b, initial_condition, func, label, color, alpha, plot_label):
    x_estimate_list, t_list = [], []
    x_previous = initial_condition
    t_previous = a
    int_sum=0
    while a < b+stepsize:
        x_next = x_previous + stepsize * func(x_previous, t_previous)
        x_estimate_list.append(x_previous)
        t_list.append(a)
        a += stepsize

        int_sum+= (x_next+x_previous)/2 * stepsize

        x_previous = x_next
        t_previous += stepsize

    print(label, stepsize, int_sum)
    plt.ylim(-5, 5)
    plt.xlim(0,3)

    if plot_label==True:
        plt.plot(t_list, x_estimate_list, label=label, alpha=alpha, color=color)
    else:
        plt.plot(t_list, x_estimate_list, alpha=alpha, color=color)


a, b = 0, 3
stepsize_list=[1,0.1,0.01]
alpha_list=[0.15, 0.4, 0.75]
for index in range(0,len(stepsize_list)):

    alpha = alpha_list[index]
    plot_label=False
    if index==2:
        plot_label=True

    euler(stepsize_list[index], a, b, 0, func_a, r'$\frac{dx}{dt}=1$, $x(0)=0$', 'r', alpha, plot_label)
    euler(stepsize_list[index], a, b, -4, func_b, r'$\frac{dx}{dt}=2t, x(0)=-4$', 'b', alpha, plot_label)
    euler(stepsize_list[index], a, b, 4, func_c, r'$\frac{dx}{dt}=-x, x(0)=4$', 'g', alpha, plot_label)


def func1(t):
    return t

def func2(t):
    return (t**2)-4

def func3(t):
    return 4*(np.e**(-t))


def analytical(func, plot_label):
    x_list, t_list = [], []
    for t in np.linspace(0,3,100):
        x=func(t)
        t_list.append(t)
        x_list.append(x)

    if plot_label==True:
        plt.plot(t_list, x_list, alpha=0.5, color='black', label='analytical solution')
    else:
        plt.plot(t_list, x_list, alpha=0.5, color='black')
    

analytical(func1, plot_label=False)
analytical(func2, plot_label=False)
analytical(func3, plot_label=True)

plt.legend()
plt.savefig('3.png')
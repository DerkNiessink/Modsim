import numpy as np
import matplotlib.pyplot as plt

"""
3
"""
def func_a(x, t):
    return 1

def func_b(x, t):
    return 2*t

def func_c(x,t):
    return -x

def euler_plot(stepsize, a, b, initial_condition, func, label, color, alpha, plot_label):
    """
    Creates a plot according to the euler approximation given the stepsizem a range 
    (a,b), an initial condition and a function. Further is it possible to specify
    the label, color, transparancy and if you want the labels plotted or not.
    """
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
    
    pass


def func1(t):
    return t

def func2(t):
    return (t**2)-4

def func3(t):
    return 4*(np.e**(-t))


def analytical(func, plot_label):
    """
    Plots the exact solution of a given function.
    """
    x_list, t_list = [], []
    for t in np.linspace(0,3,100):
        x=func(t)
        t_list.append(t)
        x_list.append(x)

    if plot_label==True:
        plt.plot(t_list, x_list, alpha=0.5, color='black', label='analytical solution')
    else:
        plt.plot(t_list, x_list, alpha=0.5, color='black')

def exercise_3(): 
    """
    Loops through different stepsizes and saves a plot for exercise 3
    """
    a, b = 0, 3
    stepsize_list=[1,0.1,0.01]
    alpha_list=[0.15, 0.4, 0.75]
    for index in range(0,len(stepsize_list)):

        alpha = alpha_list[index]
        plot_label=False
        if index==2:
            plot_label=True

        
        euler_plot(stepsize_list[index], a, b, 0, func_a, r'$\frac{dx}{dt}=1$, $x(0)=0$', 'r', alpha, plot_label)
        euler_plot(stepsize_list[index], a, b, -4, func_b, r'$\frac{dx}{dt}=2t, x(0)=-4$', 'b', alpha, plot_label)
        euler_plot(stepsize_list[index], a, b, 4, func_c, r'$\frac{dx}{dt}=-x, x(0)=4$', 'g', alpha, plot_label)

    analytical(func1, plot_label=False)
    analytical(func2, plot_label=False)
    analytical(func3, plot_label=True)

    plt.xlabel("t", fontsize=16)
    plt.ylabel("x(t)", fontsize=16)
    plt.legend()
    plt.savefig('figures/3.png')

    pass

exercise_3()
plt.clf()

"""
3 bonus
"""
def runge_kutta_2(stepsize, a, b, initial_condition, func, label, color, alpha, plot_label):
    """
    Creates a plot according to the 2nd order Runge Kutta approximation given the stepsize, 
    a range (a,b), an initial condition and a function. Further is it possible to specify
    the label, color, transparancy and if you want the labels plotted or not.
    """
    x_estimate_list, t_list = [], []
    x_previous = initial_condition
    t_previous = a
    int_sum=0
    while a < b+stepsize:
        k1=stepsize*func(x_previous, t_previous)
        k2=stepsize*func(x_previous+1/2, t_previous + (k1+stepsize)/2 )
        x_next = x_previous + k2
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
    
    pass

def exercise_3_bonus():
    """
    Loops through different stepsizes and saves a plot for the bonus of exercise 3.
    """ 
    a, b = 0, 3
    stepsize_list=[1,0.1,0.01]
    alpha_list=[0.15, 0.4, 0.75]
    for index in range(0,len(stepsize_list)):

        alpha = alpha_list[index]
        plot_label=False
        if index==2:
            plot_label=True

        
        runge_kutta_2(stepsize_list[index], a, b, 0, func_a, r'$\frac{dx}{dt}=1$, $x(0)=0$', 'r', alpha, plot_label)
        runge_kutta_2(stepsize_list[index], a, b, -4, func_b, r'$\frac{dx}{dt}=2t, x(0)=-4$', 'b', alpha, plot_label)
        runge_kutta_2(stepsize_list[index], a, b, 4, func_c, r'$\frac{dx}{dt}=-x, x(0)=4$', 'g', alpha, plot_label)

    analytical(func1, plot_label=False)
    analytical(func2, plot_label=False)
    analytical(func3, plot_label=True)

    plt.xlabel("t", fontsize=16)
    plt.ylabel("x(t)", fontsize=16)
    plt.legend()
    plt.savefig('figures/3_bonus.png')

    pass

exercise_3_bonus()

"""
5c
"""
plt.clf()
def rabbit_func(r,k,t,N_i):
    x = N_i * ( np.e**((r-k)*t) )
    return x

def plot(r,k,N_i):
    t_list = np.linspace(0,30,1000)
    x_list=[]
    for t in t_list:
        x = rabbit_func(r,k,t,N_i)
        x_list.append(x)
    plt.plot(t_list, x_list, label=f"r = {r}, k = {k}")
    plt.grid(visible=True)
    plt.xlim(0,30)
    plt.ylim(-5,200)

    plt.xlabel("t", fontsize=16)
    plt.ylabel("x(t)", fontsize=16)

    return x_list


x_list1 = plot(1, 1, 25)
x_list2 = plot(1, 2, 25)
x_list3 = plot(2,1,25)

plt.legend()
plt.savefig("figures/5cd.png")

def fixed_point(xlist):
    """
    Calculates the fixed point in a list, if there is a difference
    smaller than 0.005 between to iterations of a list it will accept 
    the point as a fixed point.
    """
    for i in range(0,len(xlist)-1):
        if round(xlist[i]-xlist[i+1], 2) == 0:
            return i

print(fixed_point(x_list1))
print(fixed_point(x_list2))
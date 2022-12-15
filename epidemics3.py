import matplotlib.pyplot as plt

def func_dIdt(b, I, S):
    return ( 1 - ( (1-b) ** I) ) * S

def func_dSdt(b, I, S):
    return -func_dIdt(b, I, S)

def euler_plot(
    stepsize, t, t_f, i, k, N, initial_condition1, func1, initial_condition2, func2, label, color='b', alpha=1
):
    """
    Creates a plot according to the euler approximation given the stepsize, a range
    (a,b), an initial condition and a function. Further is it possible to specify
    the label, color, transparancy and if you want the labels plotted or not.
    """
    I_estimate_list, S_estimate_list, t_list = [], [], []
    I_previous = initial_condition1
    S_previous = initial_condition2
    b= (1 - (1 - i)**(k/N))
    while t < t_f + stepsize:
        I_next = I_previous + stepsize * func1(b, I_previous, S_previous)
        S_next = S_previous + stepsize * func2(b, I_previous, S_previous)

        I_estimate_list.append(I_previous)
        S_estimate_list.append(S_previous)
        t_list.append(t)

        I_previous = I_next
        S_previous = S_next
        t += stepsize

    IN_list=[]
    for I in I_estimate_list:
        IN_list.append(I/N)
    

    plt.plot(t_list, IN_list, label=label, alpha=alpha, color=color)

    pass

N=10**5
i= 0.01
k = 5
euler_plot(0.001, 0, 200, 0.01, 5, N, 0.001*N, func_dIdt, 0.999*N, func_dSdt, 'Euler estimate', color='blue')
euler_plot(0.001, 0, 200, 0.1, 0.8, N, 0.001*N, func_dIdt, 0.999*N, func_dSdt, 'Euler estimate', color='orange')
plt.show()
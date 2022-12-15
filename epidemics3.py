import matplotlib.pyplot as plt
from epidemics1 import SI_model
from epidemics1 import ex_simulation


def func_dIdt(b, I, S):
    return (1 - ((1 - b) ** I)) * S


def func_dSdt(b, I, S):
    return -func_dIdt(b, I, S)


def euler_plot(
    stepsize,
    t,
    t_f,
    i,
    k,
    N,
    initial_condition1,
    func1,
    initial_condition2,
    func2,
    label,
    color="b",
    alpha=1,
):
    """
    Creates a plot according to the euler approximation given the stepsize, a range
    (a,b), an initial condition and a function. Further is it possible to specify
    the label, color, transparancy and if you want the labels plotted or not.
    """
    I_estimate_list, S_estimate_list, t_list = [], [], []
    I_previous = initial_condition1
    S_previous = initial_condition2
    b = 1 - (1 - i) ** (k / N)
    while t < t_f + stepsize:
        I_next = I_previous + stepsize * func1(b, I_previous, S_previous)
        S_next = S_previous + stepsize * func2(b, I_previous, S_previous)

        I_estimate_list.append(I_previous)
        S_estimate_list.append(S_previous)
        t_list.append(t)

        I_previous = I_next
        S_previous = S_next
        t += stepsize

    IN_list = []
    for I in I_estimate_list:
        IN_list.append(I / N)

    plt.plot(t_list, IN_list, label=label, alpha=alpha, color=color)

    pass


N = 10**5
i = 0.01
k = 5

plt.figure()
euler_plot(
    stepsize=0.001,
    t=0,
    t_f=200,
    i=0.01,
    k=5,
    N=N,
    initial_condition1=0.1 * N,
    func1=func_dIdt,
    initial_condition2=0.9 * N,
    func2=func_dSdt,
    label="Euler: k = 5, i = 0.01",
    color="green",
)
euler_plot(
    stepsize=0.001,
    t=0,
    t_f=200,
    i=0.1,
    k=0.8,
    N=N,
    initial_condition1=0.1 * N,
    func1=func_dIdt,
    initial_condition2=0.9 * N,
    func2=func_dSdt,
    label="Euler: k = 0.8, i = 0.1",
    color="red",
)


model1 = SI_model(N=10**5, k=5, i=0.01, init_i=0.1, network="random")
model2 = SI_model(N=10**5, k=0.8, i=0.1, init_i=0.1, network="random")

t_steps = 200
reps = 1


ex_simulation(model1, reps, t_steps)
ex_simulation(model2, reps, t_steps)
plt.xlabel("t", fontsize=14)
plt.ylabel(r"$\frac{I}{N}$", fontsize=16)
plt.legend()
plt.savefig("Ep_3a.png")

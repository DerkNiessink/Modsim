import numpy as np


def euler(f, t0, y0, t1, n):
    """
    Approximate the solution of a coupled system of ODEs using the Euler method.

    Arguments:
    f -- a function that takes two arguments (t, y), where t is the current time and
        y is a list of the current values of all the variables in the system
    t0 -- the initial time
    y0 -- a list of the initial values of all the variables in the system
    t1 -- the final time
    n -- the number of steps to take

    Returns:
    A list of tuples (t, y) containing the approximate solution of the ODE at
    each time step.
    """

    solution = [(t0, y0)]
    h = (t1 - t0) / n

    for i in range(1, n + 1):
        t = t0 + i * h
        y = y0 + h * f(t, y0)

        solution.append((t, y))

        y0 = y

        return solution

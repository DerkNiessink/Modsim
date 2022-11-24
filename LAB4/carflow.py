"""
"Ca.py"

Developed by:
Jenna de Vries and Derk Niessink

This module contains the class "CarFlow":

- "CarFlow": class for running a 1D celullar automaton (CA) as a one-lane
freeway model (k=2, r=1). The class allows for testing the car flow as function
of the car density and the correctness of the model as function of the number
of timesteps.
"""


from ca import CASim

import numpy as np
import matplotlib.pyplot as plt


class CarFLow:
    def __init__(self, width, height):
        self.sim = CASim()
        self.sim.width = width
        self.sim.height = height
        self.size = width * height

    def run(self):
        """Run the 1D cellular automaton"""
        t = 0
        self.sim.reset()
        while t < self.sim.height:
            self.sim.step()
            t += 1

    def carflow(self):
        """Compute the car flow and its standard deviation"""
        carflows = []
        for _ in range(self.N):
            self.run()
            cars = 0
            for row in self.sim.config:
                if row[0] == 0 and row[-1] == 1:
                    cars += 1
            carflows.append(cars / self.sim.height)
        return np.mean(carflows), np.std(carflows)

    def sweep_density(self, stepsize: float, N: int):
        """Compute the car flow with errors as function of the car density for
        a specific stepsize and number of repetitions N."""

        self.all_carflows = []
        self.carflow_errors = []
        self.densities = []
        self.N = N

        for density in np.arange(0, 1 + stepsize, stepsize):
            self.sim.init_row_prob = density
            carflow, error = self.carflow()
            self.all_carflows.append(carflow)
            self.carflow_errors.append(error)
            self.densities.append(density)

    def sweep_time(
        self,
        real_carflows: list,
        real_densities: list,
        time_range: list,
        time_stepsize: float,
        N: int,
        repetitions: int,
    ):
        """Compute the correctness of a model for a given range of times
        (heights) by comparing each critical density value with the value of a
        given reference model.

        The critical density is defined as the density where the phase
        transition occurs (increasing to suddenly decreasing car flow).

        The model is "correct" when the critical density falls within the range
        reference critical density +/- 0.05.
        """

        self.probabilities = []
        self.times = []

        critical_density = self.linear_interpolation(real_carflows, real_densities)

        for T in np.arange(time_range[0], time_range[1], time_stepsize):
            print(f"Computing T = {T}...")
            self.sim.height = T
            correct = 0

            for _ in range(repetitions):
                self.sweep_density(stepsize=0.05, N=N)
                guessed_density = self.linear_interpolation(
                    self.all_carflows, self.densities
                )

                if critical_density - 0.05 < guessed_density < critical_density + 0.05:
                    correct += 1

            self.probabilities.append(correct / repetitions)
            self.times.append(T)

    def linear_interpolation(self, carflows: list, densities: list):
        """Returns quadratric interpolated maximum given the x (densities) and y
        (carflows) lists"""
        max_index = carflows.index(max(carflows))

        y1 = carflows[max_index - 1]
        y2 = carflows[max_index]
        y3 = carflows[max_index + 1]

        x2 = densities[max_index]
        x3 = densities[max_index + 1]

        difference = 0.5 * ((y1 - y3) / (y1 - 2 * y2 + y3))
        return x2 + (x3 - x2) * difference

    def plot_carflow(self):
        plt.clf()
        plt.errorbar(
            self.densities, self.all_carflows, yerr=self.carflow_errors, fmt="."
        )
        plt.xlabel("Car density")
        plt.ylabel("Average car flow (cars / unit time)")
        plt.title(
            f"Average car flow for sample size N = {self.N}, width = {self.sim.width} and height = {self.sim.height}"
        )
        plt.savefig("carflow.png")

    def plot_probabilities(self):
        plt.clf()
        plt.plot(self.times, self.probabilities, ".")
        plt.title(
            "Probabilities of inferring the correct critical density as a\nfunction of T (height)"
        )
        plt.xlabel("T")
        plt.ylabel("Probability correct")
        plt.savefig("probabilities.png")


if __name__ == "__main__":
    test = CarFLow(width=50, height=1000)
    test.sweep_density(stepsize=0.05, N=10)
    test.plot_carflow()
    test.sweep_time(
        real_carflows=test.all_carflows,
        real_densities=test.densities,
        time_range=[2, 35],
        time_stepsize=1,
        N=10,
        repetitions=20,
    )
    test.plot_probabilities()

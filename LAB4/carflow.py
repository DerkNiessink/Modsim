from ca import CASim

import numpy as np
import matplotlib.pyplot as plt


def linear_interpolation(carflows, densities):
        """Returns quadratric interpolated maximum given the x (densities) and y
        (carflows) lists"""
        max_index=carflows.index(max(carflows))

        y1 = carflows[max_index-1]
        y2 = carflows[max_index]
        y3 = carflows[max_index+1]

        x2 = densities[max_index]
        x3 = densities[max_index+1]

        difference= 0.5 * ( (y1-y3) / (y1-2*y2+y3))
        x = x2+ (x3-x2)*difference
        return x


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
        carflows = []
        for _ in range(self.N):
            self.run()
            cars = 0
            for row in self.sim.config:
                if row[0] == 0 and row[-1] == 1:
                    cars += 1
            carflows.append(cars / self.sim.height)
        return np.mean(carflows), np.std(carflows)

    def sweep_density(self, stepsize, N):

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

    def sweep_time(self, real_carflows, real_densities):

        self.probabilities = []
        self.times = []

        critical_density = linear_interpolation(real_carflows, real_densities)
        print(critical_density)
        for T in np.arange(3, 25, 1):
            self.sim.height = T
            correct=0
            for _ in range(20):
                self.sweep_density(0.05, 10)
                guessed_density = linear_interpolation(self.all_carflows, self.densities)
                print(T, guessed_density)
                if (
                    critical_density - 0.05
                    < guessed_density
                    < critical_density + 0.05
                ):
                    correct += 1

            self.probabilities.append(correct / 10)
            self.times.append(T)

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
        plt.xlabel("T")
        plt.ylabel("Probability correct")
        plt.savefig("probabilities.png")


if __name__ == "__main__":
    test = CarFLow(width=50, height=500)
    test.sweep_density(stepsize=0.05, N=10)
    test.plot_carflow()
    test.sweep_time(test.all_carflows, test.densities)
    test.plot_probabilities()

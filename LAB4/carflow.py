from ca import CASim

import numpy as np
import matplotlib.pyplot as plt


class CarFLow:
    def __init__(self, width, height):
        self.sim = CASim()
        self.sim.width = width
        self.sim.height = height
        self.size = width * height
        self.all_carflows = []
        self.carflow_errors = []
        self.densities = []

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
        self.N = N
        for density in np.arange(0, 1 + stepsize, stepsize):
            self.sim.init_row_prob = density
            carflow, error = self.carflow()
            self.all_carflows.append(carflow)
            self.carflow_errors.append(error)
            self.densities.append(density)

    def plot(self):
        plt.errorbar(
            self.densities, self.all_carflows, yerr=self.carflow_errors, fmt="."
        )
        plt.xlabel("Car density")
        plt.ylabel("Average car flow (cars / unit time)")
        plt.title(
            f"Average car flow for sample size N = {self.N}, width = {self.sim.width} and height = {self.sim.height}"
        )
        plt.savefig("carflow.png")


if __name__ == "__main__":
    test = CarFLow(width=50, height=1000)
    test.sweep_density(stepsize=0.05, N=5)
    test.plot()

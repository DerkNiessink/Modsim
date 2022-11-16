from ca import CASim
import numpy as np
from ca import configs

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class CALangtonTest:
    def __init__(self, width, height, k, r, init_row_prob):
        self.sim = CASim()
        self.sim.init_row_prob = init_row_prob
        self.sim.width = width
        self.sim.height = height
        self.sim.k = k
        self.sim.r = r

    def random_table(self, lamb):
        self.all_configs = configs(self.sim.r, self.sim.k)
        self.size = width * height
        self.average_shannons = []
        self.errors = []
        self.lambdas = []

    def run(self, lamb):
        t = 0
        self.sim.reset(langton=True, lamb=lamb)
        while t < self.sim.height:
            self.sim.step_rt()
            t += 1

    def table_walk_through(self, lamb):
        t = 0
        self.sim.reset(langton=True, lamb=lamb)
        while t < self.sim.height:
            self.sim.step_twt()
            t += 1


    def sweep_langton(self, N):
        lambda_steps=np.linspace(0,1,num=10)
        for lamb in lambda_steps:
            self.random_table(lamb)
    
    def sweep_langton(self, N, steps):
        for lamb in np.arange(0.01, 1, 1 / steps):
            shannons = []
            for _ in range(N):
                self.run(lamb)
                shannons.append(self.shannon(self.sim.config))

            self.average_shannons.append(np.mean(shannons))
            self.errors.append(np.std(shannons) / np.sqrt(len(shannons)))
            self.lambdas.append(lamb)

    def shannon(self, config):
        count_dict = self.count_neighboorhood_configs(config)

        shannon = 0
        for key in count_dict:
            prob = count_dict[key] / self.size
            shannon += -prob * np.log2(prob)
        return shannon

    def count_neighboorhood_configs(self, config):
        neighborhood_configs = []
        for state in config.tolist():

            # Take boundary conditions in account
            state.insert(0, state[-1])
            state.append(state[1])

            neighborhood_configs.extend(
                [(state[i], state[i + 1], state[i + 2]) for i in range(len(state) - 2)]
            )

        return dict(Counter(neighborhood_configs))

    def plot(self):
        figure = plt.figure(figsize=(10, 10))

        plt.ylabel("Average shannon entropy")
        plt.xlabel("$\lambda$")
        plt.ylim(0, 2.5)

        plt.errorbar(
            self.lambdas,
            self.average_shannons,
            yerr=self.errors,
            fmt=".",
        )
        figure.savefig("langton")


if __name__ == "__main__":
    test = CALangtonTest(width=50, height=100, k=2, r=1, init_row_prob=0.9)
    test.sweep_langton(N=60, steps=50)
    test.plot()

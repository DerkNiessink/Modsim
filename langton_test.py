from ca import CASim
from ca import configs

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class CALangtonTest:
    def __init__(self, width, height, k, r):
        self.sim = CASim()
        self.sim.width = width
        self.sim.height = height
        self.sim.k = k
        self.sim.r = r
        self.sim.rule_set_size = k ** (2 * r + 1)
        self.all_configs = configs(self.sim.r, self.sim.k)
        self.size = width * height
        self.sim.init_twt()

    def run(self, mode, lamb=None, new_rule_test=True):
        t = 0
        self.sim.reset(mode, lamb, new_rule_test)
        while t < self.sim.height:
            self.sim.step()
            t += 1

    def sweep_langton_rt(self, N, steps):
        """Run the simulation for each lambda between 0 and 1 and with
        specified number of steps, N times. Compute the average shannon entropy
        for each of the N different random intitial confgurations."""

        self.empty_lists()
        for lamb in np.arange(0.01, 1, 1 / steps):
            shannons = []
            for _ in range(N):
                self.run(mode="random", lamb=lamb)
                shannons.append(self.shannon(self.sim.config))

            self.add_lists(shannons, lamb)

    def sweep_langton_twt(self, N):

        self.empty_lists()
        while len(self.sim.twt_index_list) != 0 and self.sim.twt_lamb <= 1:
            shannons = []
            self.run(mode="walkthrough", new_rule_test=True)
            for _ in range(N):
                self.run(mode="walkthrough", new_rule_test=False)
                shannons.append(self.shannon(self.sim.config))

            self.add_lists(shannons, self.sim.twt_lamb)

    def empty_lists(self):
        self.average_shannons = []
        self.errors = []
        self.lambdas = []

    def add_lists(self, shannons, lamb):
        self.average_shannons.append(np.mean(shannons))
        self.errors.append(np.std(shannons) / np.sqrt(len(shannons)))
        self.lambdas.append(lamb)

    def shannon(self, config):
        """Compute the average shannon entropy over all states (i.e row in
        simulation config)."""

        count_dict = self.count_neighboorhood_configs(config)
        shannon = 0
        for key in count_dict:
            prob = count_dict[key] / self.size
            shannon += -prob * np.log2(prob)
        return shannon

    def count_neighboorhood_configs(self, config):
        """Count each of the possible neighboorhood configurations. A
        "neighborhood" consists of three sequential states and is effected by
        the periodic boundary condition.

        Example:
        config = [[1 1 1 0] [0 1 1 0]] returns
        {(1 1 1): 1, (1 1 0): 2, (1 0 1): 1, (0 1 1): 2, (1 0 0): 1, (0 0 1): 1}
        """

        neighborhood_configs = []
        for state in config.tolist():

            # Take boundary conditions in account
            state.insert(0, state[-1])
            state.append(state[1])

            neighborhood_configs.extend(
                [(state[i], state[i + 1], state[i + 2]) for i in range(len(state) - 2)]
            )

        return dict(Counter(neighborhood_configs))

    def plot(self, file: str):

        figure = plt.figure(figsize=(10, 10))
        plt.ylabel("Average shannon entropy")
        plt.xlabel("$\lambda$")
        plt.errorbar(
            self.lambdas,
            self.average_shannons,
            yerr=self.errors,
            fmt=".",
        )
        figure.savefig(file)


if __name__ == "__main__":
    test = CALangtonTest(width=64, height=64, k=4, r=1)

    test.sweep_langton_rt(N=20, steps=30)
    test.plot("langton_rt")

    test.sweep_langton_twt(N=20)
    test.plot("langton_twt")

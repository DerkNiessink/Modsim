import numpy as np
import matplotlib.pyplot as plt

from ca import CASim


class CASimTest:
    def __init__(self, width, height, k, r, init_row_prob):
        self.sim = CASim()
        self.sim.init_row_prob = init_row_prob
        self.sim.width = width
        self.sim.height = height
        self.sim.k = k
        self.sim.r = r
        self.max_rule_num = self.sim.k ** (self.sim.k ** (2 * self.sim.r + 1))
        self.average_cycle_lengths = []

    def run(self):
        t = 0
        self.sim.reset()
        while t < self.sim.height:
            self.sim.step()
            t += 1

    def sweep_rule(self, N):
        """Run the simulation for each rule, N times. Compute the average cycle
        length for each of the N different random intitial confgurations."""

        for rule in range(self.max_rule_num):
            self.sim.rule = rule

            cycle_lengths = []
            for _ in range(N):
                self.run()
                index_dict = self.make_index_dict()
                cycle_lengths.append(self.cycle_length(index_dict))

            self.average_cycle_lengths.append(np.mean(cycle_lengths))

    def make_index_dict(self):
        """Make dictionary with a set of rows as keys and their indexes in the
        simulation configuration as values.

        Example:
        The configuration array [[1 1 1] [1 0 0] [1 1 1] [1 1 1] [1 0 0]] will yield
        index_dict = {(1, 1, 1): [0, 2, 3], (1, 0, 0): [1, 4]}"""

        config_list = [tuple(config) for config in self.sim.config.tolist()]

        index_dict = {}
        for row, index in enumerate(config_list):
            # Make a new row key in the dictionary or append the
            # index if the key already existed
            index_dict.setdefault(index, []).append(row)

        return index_dict

    def cycle_length(self, index_dict):
        """Calculate the average cycle length of a simulation configuration by
        calculating the differences of the indexes for each row and taking the
        average."""

        cycle_lengths = []
        for _, indexes in index_dict.items():
            # if the row was an unique row in the simulation configuration, take
            # the height as the max cycle length.
            if len(indexes) == 1:
                cycle_lengths.append(self.sim.height)
            else:
                diffs = np.diff(np.array(indexes))
                cycle_lengths.extend(diffs)

        return np.mean(cycle_lengths)

    def plot(self, cycle_lengths):
        rules = [rule for rule in range(self.max_rule_num)]
        figure = plt.figure()
        plt.ylabel("Average cycle length")
        plt.xlabel("Rule number (k)")
        plt.title(
            f"""Average cycle lengths for the 1D CA (k = {self.sim.k}, r = {self.sim.r})
            for width = {self.sim.width} and height = {self.sim.height}.""",
            loc="left",
        )
        plt.xlim(-1, self.max_rule_num + 1)
        plt.plot(rules, cycle_lengths, ".")
        figure.savefig("cycle_lengths")
        plt.show()


if __name__ == "__main__":
    sim_test = CASimTest(width=40, height=100, k=2, r=1, init_row_prob=0.4)
    sim_test.sweep_rule(N=25)
    sim_test.plot(sim_test.average_cycle_lengths)

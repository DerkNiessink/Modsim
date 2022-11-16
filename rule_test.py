from ca import CASim

import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.patches import Patch


class CASimFig:
    def __init__(self, width, height, k, r):
        self.sim = CASim()
        self.sim.width = width
        self.sim.height = height
        self.sim.k = k
        self.sim.r = r
        self.max_rule_num = self.sim.k ** (self.sim.k ** (2 * self.sim.r + 1))
        self.average_cycle_lengths = []
        self.errors = []

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

            # Use the standard error of the mean (SEM) as average cycle length
            # errors, because cycle_lengths is a distribution of means.
            self.errors.append(np.std(cycle_lengths) / np.sqrt(len(cycle_lengths)))

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

    def csv_lst():
        csv_filename = "rule_class_wolfram.csv"
        with open(csv_filename) as f:
            reader = csv.reader(f)
            lst = list(reader)
        return lst

    def plot(self):
        rules = [rule for rule in range(self.max_rule_num)]
        figure = plt.figure()

        lst = CASimFig.csv_lst()
        color_list = ["purple", "blue", "red", "green", "purple"]

        color_index_list = []
        for pair in lst:
            ca_code = eval(pair[1])
            color_index_list.append(color_list[ca_code])

        plt.ylabel("Average cycle length")
        plt.xlabel("Rule number")
        plt.title(
            f"""Average cycle lengths for the 1D CA (k = {self.sim.k}, r = {self.sim.r})
            for width = {self.sim.width} and height = {self.sim.height}.""",
            loc="left",
        )
        plt.xlim(-1, self.max_rule_num + 1)

        plt.errorbar(
            rules,
            self.average_cycle_lengths,
            yerr=self.errors,
            fmt=".",
            ecolor=color_index_list,
            mfc="white",
            mec="white",
        )

        plt.scatter(rules, self.average_cycle_lengths, c=color_index_list, marker="o")

        legend_elements = [
            Patch(facecolor=color, edgecolor="w")
            for color in ["blue", "red", "green", "purple"]
        ]

        plt.legend(
            handles=legend_elements,
            labels=[f"Class 1", f"Class 2", f"Class 3", f"Class 4"],
            loc="upper left",
            bbox_to_anchor=[1, 1],
        )
        plt.figure(figsize=(10, 10))
        figure.savefig("cycle_lengths", bbox_inches="tight")


if __name__ == "__main__":
    # Test paramaters (including N) can be adjusted.
    sim_test = CASimFig(width=10, height=5, k=2, r=1)
    sim_test.sweep_rule(N=2)
    sim_test.plot()

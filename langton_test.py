"""
"langton_test.py"

Developed by:
Jenna de Vries and Derk Niessink

This module contains the class "CALangtonTest":

"CALangtonTest" is a class for testing a 1D celullar automaton (CA) using the
Langton parameter. Both the "random table" and "table-walkthrough" techniques for
building the rule table are implemented and can be executed with respectively the
"sweep_langton_rt" and the "sweep_langton_twt" method.

The class allows plotting the Shannon entropy against the langton parameters
with the "plot" method and are saved as "langton_rt.png" and "langton_twt.png"
for respectively the "random_table" and the "table-walkthrough" method.

For test parameters k=2 and r=1 and technique "random-table", a colored figure
with colors following the Wolframm classes from "rule_class_wolfram.csv" can
be plotted using the "plot-colors" method.
"""


from ca import CASim
from ca import configs
from ca import base_k_to_decimal

from matplotlib.patches import Patch
from collections import Counter
import numpy as np
import csv
import matplotlib.pyplot as plt


def csv_lst(fn: str):
    with open(fn) as f:
        reader = csv.reader(f)
        lst = list(reader)
    return lst


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
        """Run the 1D cellular automaton"""
        t = 0
        self.sim.reset(mode, lamb, new_rule_test)
        while t < self.sim.height:
            self.sim.step()
            t += 1

    def sweep_langton_rt(self, N, steps):
        """Run the simulation for each lambda between 0 and 1 using the random
        table method and with specified number of steps, N times.

        Compute the average shannon entropy for each of the N different random
        intitial confgurations."""

        self.empty_lists()
        for lamb in np.arange(0.01, 1, 1 / steps):
            shannons = []
            for _ in range(N):
                self.run(mode="random", lamb=lamb)
                shannons.append(self.shannon(self.sim.config))

            self.add_lists(shannons, lamb, self.sim.rt_rule_set)

    def sweep_langton_twt(self, N):
        """Run the simulation using the table-walkthrough method:
        Start with a rule set consisting of only the quiescent state (0) and
        changing one rule to a random state (!= quiescent sate) every N runs.

        Compute the average Shannon entropy for each of the N different random
        intitial confgurations."""

        self.empty_lists()
        while len(self.sim.twt_index_list) != 0 and self.sim.twt_lamb <= 1:
            shannons = []
            self.run(mode="walkthrough", new_rule_test=True)
            for _ in range(N):
                self.run(mode="walkthrough", new_rule_test=False)
                shannons.append(self.shannon(self.sim.config))

            self.add_lists(shannons, self.sim.twt_lamb, self.sim.twt_rule_set)

    def empty_lists(self):
        self.average_shannons = []
        self.errors = []
        self.lambdas = []
        self.rule_numbers = []

    def add_lists(self, shannons, lamb, rule_set):
        self.average_shannons.append(np.mean(shannons))
        self.errors.append(np.std(shannons) / np.sqrt(len(shannons)))
        self.lambdas.append(lamb)
        self.rule_numbers.append(base_k_to_decimal(rule_set, self.sim.k, self.sim.r))

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
        plt.ylabel("Average H", fontsize=15)
        plt.xlabel("$\lambda$", fontsize=15)
        plt.errorbar(
            self.lambdas,
            self.average_shannons,
            yerr=self.errors,
            fmt=".",
        )
        plt.title(
            f"""Average Shannon entropy for the 1D CA (k = {self.sim.k}, r = {self.sim.r})
            for width = {self.sim.width} and height = {self.sim.height}.""",
            loc="center",
            fontsize=15,
        )
        figure.savefig(file)

    def plot_colors(self):
        """For this plot function only use k=2 r=1, the given Wolfram class list
        only includes the rule sets for these parameters. The max rule set is
        255."""

        figure = plt.figure(figsize=(10, 10))

        color_list = ["purple", "blue", "red", "green", "purple"]
        ca_code_list = []
        color_list_matched = []
        ca_code_list_matched = []
        lst = csv_lst("rule_class_wolfram.csv")

        for pair in lst:
            ca_code = eval(pair[1])
            ca_code_list.append(ca_code)

        for rule in self.rule_numbers:
            ca_code = ca_code_list[rule]
            ca_code_list_matched.append(ca_code)
            color_list_matched.append(color_list[(ca_code)])

        plt.ylabel("Average H", fontsize=15)
        plt.xlabel("$\lambda$", fontsize=15)
        plt.title(
            f"""Average Shannon entropy for the 1D CA (k = {self.sim.k}, r = {self.sim.r})
            for width = {self.sim.width} and height = {self.sim.height}.""",
            loc="center",
            fontsize=15,
        )
        plt.errorbar(
            self.lambdas,
            self.average_shannons,
            yerr=self.errors,
            fmt=".",
            ecolor=color_list_matched,
            mfc="black",
            mec="black",
        )

        plt.scatter(
            self.lambdas, self.average_shannons, c=color_list_matched, marker="o"
        )

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
        figure.savefig("langton_rt", bbox_inches="tight")


if __name__ == "__main__":
    # Test paramaters (including N and steps) can be adjusted.
    test = CALangtonTest(width=64, height=64, k=4, r=1)

    test.sweep_langton_rt(N=30, steps=30)
    test.plot("langton_rt")

    test.sweep_langton_twt(N=10)
    test.plot("langton_twt")

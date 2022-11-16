"""
"Ca.py"

Developed by:
Jenna de Vries and Derk Niessink

This module contains two classes:

- "CASim": class for running a 1D celullar automaton (CA) with a GUI.

- "CASimFig" class for measuring and plotting the average cycle length per rule.

Run this module as "Ca.py gui" to start up the gui. Running this module just as
"Ca.py" will run the simulation for all rules and generate the plot in the file:
"cycle_lengths.png". The parameters of the CA can be adjusted in the bottom of
this file. The plot is colored in with data from the file: "rule_class_wolfram.csv".
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import random
from matplotlib.patches import Patch
import csv
import sys
from pyics import Model


def decimal_to_base_k(n, k, r):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""

    new_base = []
    while n > 0:
        new_base.insert(0, n % k)
        n = int(n / k)

    while len(new_base) < (2 * r + 1):
        new_base.insert(0, 0)

    return new_base


def configs(r, k):
    """Create list with all possible configurations for a specific base-k and
    range r."""
    max_index = k ** (2 * r + 1) - 1
    return [decimal_to_base_k(i, k, r) for i in np.arange(max_index, -1, -1)]


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.init_row_prob = 0.9
        self.rule_dict = {}
        self.config = None

        self.make_param("r", 1)
        self.make_param("k", 2)
        self.make_param("width", 200)
        self.make_param("height", 200)
        self.make_param("rule", 5, setter=self.setter_rule)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k**rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""

        new_base = decimal_to_base_k(self.rule, self.k, self.r)
        zeros = self.k ** (2 * self.r + 1) - len(new_base)
        rule_set = [0 for _ in range(zeros)] + new_base
        configurations = configs(self.r, self.k)

        for config, rule in zip(configurations, rule_set):
            self.rule_dict[f"{config}"] = rule

    def build_langton_set_rt(self, lamb):
        length = self.k**(2*self.r+1)
        rule_set = []
        for i in range(length):
            g=np.random.rand()
            if g>lamb:
                rule_set.append(0)
            else:
                rule_set.append(np.random.randint(1, self.k))

        configurations = configs(self.r, self.k)

        for config, rule in zip(configurations, rule_set):
            self.rule_dict[f"{config}"] = rule


    def build_langton_set_twt(self, length, rule_set, index_list):
        index = np.random.choice(index_list)
        index_list.remove(index)
        rule_set[index]=1

        configurations = configs(self.r, self.k)

        lamb = (len(rule_set) - len(index_list)) / len(rule_set)
        for config, rule in zip(configurations, rule_set):
            self.rule_dict[f"{config}"] = rule

        return lamb

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        return self.rule_dict[f"{inp}"]

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""

        initial_row = [
            1 if random() > self.init_row_prob else 0 for _ in range(self.width)
        ]
        return initial_row

    def reset(self, random=False, lamb=None, walkthrough = False):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        if random:
            self.build_langton_set_rt(lamb)

        elif walkthrough:
            length = self.k**(2*self.r+1)
            rule_set = [0] * length
            index_list=np.linspace(0,length)
            self.lamb = self.build_langton_set_twt(lamb, length, rule_set, index_list)
        else:
            self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(
            self.config,
            interpolation="none",
            vmin=0,
            vmax=self.k - 1,
            cmap=matplotlib.cm.binary,
        )
        plt.axis("image")
        plt.ylabel("time")
        plt.xlabel("position")

        colors = ["white", "black"]
        legend_elements = [
            Patch(facecolor=color, edgecolor="black") for color in colors
        ]
        plt.legend(
            handles=legend_elements,
            labels=[f"value = 0", f"value = 1"],
            loc="upper left",
            bbox_to_anchor=[1, 1],
        )

        plt.title("%d generations" % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [
                i % self.width for i in range(patch - self.r, patch + self.r + 1)
            ]
            values = self.config[self.t - 1, indices]
            values = [int(value) for value in values]
            self.config[self.t, patch] = self.check_rule(values)
    
    


class CASimFig:
    def __init__(self, width, height, k, r, init_row_prob):
        self.sim = CASim()
        self.sim.init_row_prob = init_row_prob
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

    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        sim = CASim()
        from pyics import GUI

        cx = GUI(sim)
        cx.start()

    else:
        # Test paramaters (including N) can be adjusted.
        sim_test = CASimFig(width=10, height=5, k=2, r=1, init_row_prob=0.4)
        sim_test.sweep_rule(N=2)
        sim_test.plot()

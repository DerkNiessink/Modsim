"""
"Ca.py"

Developed by:
Jenna de Vries and Derk Niessink

This module contains the class "CASim":

- "CASim": class for running a 1D celullar automaton (CA) with a GUI.
"""


import numpy as np
import matplotlib.pyplot as plt
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
        self.rule_dict = {}
        self.config = None

        self.make_param("init_row_prob", 0.5)
        self.make_param("r", 1)
        self.make_param("k", 2)
        self.make_param("width", 50)
        self.make_param("height", 50)
        self.rule = 184

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

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        return self.rule_dict[f"{inp}"]

    def setup_initial_row(self):
        """First sets initial row with all zeroes, then replaces the amount of
        wanted cars with ones."""

        initial_row = [0 for _ in range(self.width)]
        index_list = []
        index = 0
        for _ in initial_row:
            index_list.append(int(index))
            index += 1

        for _ in range(int(self.init_row_prob * self.width)):
            choice = np.random.choice(index_list)
            index_list.remove(choice)
            initial_row[choice] = 1

        return initial_row

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

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
        legend_elements = [Patch(facecolor=color, edgecolor="w") for color in colors]
        plt.legend(
            handles=legend_elements,
            labels=[f"free space", f"car"],
            loc="upper left",
            bbox_to_anchor=[1, 1],
        )
        plt.title(f"car density {self.init_row_prob}")

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


if __name__ == "__main__":

    sim = CASim()
    from pyics import GUI

    cx = GUI(sim)
    cx.start()

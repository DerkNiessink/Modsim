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
        length = self.k ** (2 * self.r + 1)
        rule_set = []
        for i in range(length):
            g = np.random.rand()
            if g > lamb:
                rule_set.append(0)
            else:
                rule_set.append(np.random.randint(1, self.k))

        configurations = configs(self.r, self.k)

        for config, rule in zip(configurations, rule_set):
            self.rule_dict[f"{config}"] = rule

    def build_langton_set_twt(self, length, rule_set, index_list):
        index = np.random.choice(index_list)
        index_list.remove(index)
        rule_set[index] = 1

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

        return [np.random.randint(0, self.k) for _ in range(self.width)]

    def reset(self, random=False, lamb=None, walkthrough=False):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        if random:
            self.build_langton_set_rt(lamb)

        elif walkthrough:
            length = self.k ** (2 * self.r + 1)
            rule_set = [0] * length
            index_list = np.linspace(0, length)
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


if __name__ == "__main__":

    sim = CASim()
    from pyics import GUI

    cx = GUI(sim)
    cx.start()

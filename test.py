import numpy as np
import matplotlib.pyplot as plt

from ca import CASim


class CASimTest:
    def __init__(self, width, height):
        self.sim = CASim()
        self.sim.width = width
        self.sim.height = height
        self.max_rule_num = self.sim.k ** (self.sim.k ** (2 * self.sim.r + 1))
        self.average_cycle_list = []

    def run(self):
        t = 0
        self.sim.reset()
        while t < self.sim.height:
            self.sim.step()
            t += 1

    def sweep_rule(self, N):
        for rule in range(self.max_rule_num):
            self.sim.rule = rule
            # Run the simulation and search for cycles for N random initial rows
            average_cycles_row = []
            for _ in range(N):
                self.run()
                cycles = self.find_cycles()
                # print(cycles, "\n", self.sim.config, "\n")
                if cycles == None:
                    average_cycles_row.append(self.sim.height)
                else:
                    average_cycles_row.append(np.mean(cycles))

            self.average_cycle_list.append(np.mean(average_cycles_row))

    def find_cycles(self):
        config_list = [tuple(config) for config in self.sim.config.tolist()]
        config_set = set(config_list)

        if len(config_set) == len(config_list):
            # This means that each row is unique, so no cycles.
            return None

        self.count = 0
        self.counting = False
        self.total_cycles = []
        # Loop through config_set and compare with list to find cycles.
        for test_config in config_set:

            self.cycles = []
            self.count = 0

            for config in config_list:

                if config == test_config:
                    self.start_stop_count()

                if self.counting == True:
                    self.count += 1

            self.add_total_cycles()
            self.counting = False

        return self.total_cycles

    def start_stop_count(self):
        # Negate counting: True -> False and False -> True
        self.counting = not self.counting

        # Append the count if True -> False, because then a cycle was found
        if self.counting == False:
            self.cycles.append(self.count)
            self.counting = True

        self.count = 0

    def add_total_cycles(self):
        if self.cycles == []:
            self.total_cycles.append(self.sim.height)
        else:
            self.total_cycles.extend(self.cycles)

    def plot(self, cycle_lengths):
        rules = [rule for rule in range(self.max_rule_num)]
        plt.ylabel("Average cycle length")
        plt.xlabel("Rule number (k)")
        plt.xlim(-1, self.max_rule_num + 1)
        plt.plot(rules, cycle_lengths, ".")
        plt.show()


if __name__ == "__main__":
    sim_test = CASimTest(width=10, height=20)
    sim_test.sweep_rule(N=10)
    print(sim_test.average_cycle_list)
    sim_test.plot(sim_test.average_cycle_list)

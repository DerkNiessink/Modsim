import numpy as np

from ca import CASim


class CASimTest:
    def __init__(self):
        self.sim = CASim()
        self.sim.width = 3
        self.sim.height = 10

    def run(self):
        t = 0
        self.sim.reset()
        while t < self.sim.height:
            self.sim.step()
            t += 1

        # print(self.sim.config)

    def sweep_rule(self):
        max_rule_num = self.sim.k ** (self.sim.k ** (2 * self.sim.r + 1))
        for rule in range(max_rule_num):
            self.sim.rule = rule
            for _ in range(2):
                self.run()
                self.find_cycle()

    def find_cycle(self):
        config_list = self.sim.config.tolist()
        print(config_set)


if __name__ == "__main__":
    sim_test = CASimTest()
    sim_test.sweep_rule()

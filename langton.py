from ca import CASim
import numpy as np


class CALangtonTest:
    def __init__(self, width, height, k, r, init_row_prob):
        self.sim = CASim()
        self.sim.init_row_prob = init_row_prob
        self.sim.width = width
        self.sim.height = height
        self.sim.k = k
        self.sim.r = r

    def run(self, lamb):
        t = 0
        self.sim.reset(langton=True, lamb=lamb)
        while t < self.sim.height:
            self.sim.step()
            t += 1

    def sweep_langton(self, N):
        for lamb in np.arange(0, 1, 1 / N):
            self.run(lamb)
            self.shannon(self.sim.config)

    def shannon(self, config):
        print(config)

    def plot(self):
        pass


if __name__ == "__main__":
    test = CALangtonTest(width=10, height=10, k=2, r=1, init_row_prob=0.5)
    test.sweep_langton(N=10)

from ca import CASim


class CALangtonTest:
    def __init__(self, width, height, k, r, init_row_prob):
        self.sim = CASim()
        self.sim.init_row_prob = init_row_prob
        self.sim.width = width
        self.sim.height = height
        self.sim.k = k
        self.sim.r = r
        self.max_rule_num = self.sim.k ** (self.sim.k ** (2 * self.sim.r + 1))

    def run(self):
        t = 0
        self.sim.reset(langton=True)
        while t < self.sim.height:
            self.sim.step()
            t += 1

    def sweep_langton(self, N):
        for rule in range(self.max_rule_num):
            self.sim.rule = rule

            for _ in range(N):
                self.run()

    def plot(self):
        pass


if __name__ == "__main__":
    test = CALangtonTest(width=5, height=10, k=2, r=1, init_row_prob=0.4)
    test.sweep_langton(N=10)

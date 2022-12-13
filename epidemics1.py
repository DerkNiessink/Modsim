import networkx as nx
import matplotlib.pyplot as plt
from random import random


class SI_model:
    def __init__(self, N, k, i, init_i):
        self.G = nx.fast_gnp_random_graph(n=N, p=k / N)
        self.i = i
        self.N = N
        self.nInfected = 0
        self.init_infected(init_i)

    def init_infected(self, i):
        for node in self.G.nodes:
            if random() <= i:
                infected = True
                self.nInfected += 1
            else:
                infected = False

            self.G.nodes[node]["infected"] = infected

    def update(self):
        infected_dict = nx.get_node_attributes(self.G, "infected")
        for node in self.G.nodes:
            if infected_dict[node] == True:
                self.infect_neighbours(node)

    def infect_neighbours(self, node):
        for neighbor in nx.all_neighbors(self.G, node):
            if not self.G.nodes[neighbor]["infected"] and random() <= self.i:
                self.G.nodes[neighbor]["infected"] = True
                self.nInfected += 1


if __name__ == "__main__":

    model = SI_model(N=10**5, k=5, i=0.01, init_i=0.1)

    time = [t for t in range(500)]
    norm_prevelances = []
    for _ in time:
        model.update()
        norm_prevelances.append(model.nInfected / model.N)

    plt.plot(time, norm_prevelances, ".")
    plt.xlabel("t", fontsize=14)
    plt.ylabel("I/N", fontsize=14)
    plt.savefig("Ep_1b.png")

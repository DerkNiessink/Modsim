import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import random


class SI_model:
    """Susceptible-Infected model for networks"""

    def __init__(
        self, N: int, k: int, i: float, init_i: float, network: str, power=None
    ):
        """
        Initialize the graph and set the initial number of infected nodes.

        N: The number of nodes in the graph.
        k: The number of edges in the graph.
        i: The infection probability.
        init_i: The initial probability that a node is infected.
        network: Can be "random" or "scale_free"
        power: Power for the powerlaw_sequence of the "scale_free" network
               with default None.
        """

        self.k = k
        self.i = i
        self.N = N
        self.G = self.setup_network(network, power)
        self.init_i = init_i
        self.reset()

    def setup_network(self, network: str, power=None):
        """
        Setup the desired network specified in "network".

        network: Can be "random" or "scale_free"
        power: Power for the powerlaw_sequence of the "scale_free network
               with default None.

        Returns the desired nx network.
        """
        if network == "random":
            return nx.fast_gnp_random_graph(n=self.N, p=self.k / self.N)

        if network == "scale_free":
            s = nx.utils.powerlaw_sequence(self.N, power)
            s = s / np.mean(s) * self.k
            return nx.expected_degree_graph(s, selfloops=False)

    def init_infected(self, i: float):
        """
        Set the initial number of infected nodes.

        i: The probability that a node is initially infected.
        """
        N_i = self.N * i
        # while self.nInfected < N_i:
        for node in self.G.nodes:
            if random() <= i:
                infected = True
                self.nInfected += 1
            else:
                infected = False

            self.G.nodes[node]["infected"] = infected

    def update(self):
        """
        Update the state of the graph by infecting the neighbors of infected
        nodes and keep track of the the normalized prevalence and the average
        node degree of the new infected nodes.
        """
        self.t += 1
        infected_dict = nx.get_node_attributes(self.G, "infected")
        for node in self.G.nodes:
            if infected_dict[node] == True:
                self.infect_neighbours(node)

        self.norm_prevalences.append(self.nInfected / self.N)
        self.time.append(self.t)
        self.avg_k_infections.append(np.mean(self.k_infections))
        self.k_infections = []

    def infect_neighbours(self, node: int):
        """
        Infect the neighbors of a given node with probability `i`.

        node: The node whose neighbors will be infected.
        """
        for neighbor in nx.all_neighbors(self.G, node):
            if not self.G.nodes[neighbor]["infected"] and random() <= self.i:
                self.G.nodes[neighbor]["infected"] = True
                self.nInfected += 1

                # Keep track of the the degree of every node that gets infected.
                self.k_infections.append(len(list(nx.all_neighbors(self.G, neighbor))))

    def reset(self):
        """
        Reset the simulation.
        """
        self.norm_prevalences = []
        self.time = []
        self.k_infections = []
        self.avg_k_infections = []
        self.nInfected = 0
        self.t = 0
        self.init_infected(self.init_i)

    def plot(self, fn: str):
        """
        Plot the normalized prevalence against time and save it as `fn`.

        fn: filename
        """
        plt.plot(self.time, self.norm_prevalences)
        plt.xlabel("t", fontsize=14)
        plt.ylabel(r"$\frac{I}{N}$", fontsize=14)
        plt.savefig(fn)


def ex_simulation(model: SI_model, reps: int, t_steps: int) -> tuple[list, list]:
    """
    Execute the simulation of the SI model for a given number of time steps and
    repetitions. Plot the average and the standard deviation of the normalized
    prevalences against time.

    reps: number of simulation repetitions.
    t_steps: number of time steps in the simulation.

    Returns a list of the time range and the average normalized prevalences in
    a tuple.
    """

    all_data = np.zeros(shape=t_steps)

    # Execute simulations for a number of reps
    for i in range(reps):

        # Execute the simulation for a number of time steps
        for _ in range(t_steps):
            model.update()

        # Save normalized prevalences
        all_data = np.vstack([all_data, model.norm_prevalences])

        # Let data of the last repetition be accessible
        if i != reps - 1:
            model.reset()

    # Delete initial row with zeros
    all_data = np.delete(all_data, (0), axis=0)

    norm_prevalences = list(np.average(all_data, axis=0))
    time = [t for t in range(t_steps)]

    errors = list(np.std(all_data, axis=0))

    plt.errorbar(
        time,
        norm_prevalences,
        yerr=errors,
        fmt="-",
        label=f"k = {model.k}, i = {model.i}",
    )

    return norm_prevalences

def R0(infections):
    R0_list, index_list=[], []
    for index in range(0, len(infections)-1):
        R0=(infections[index+1]/infections[index])-1
        R0_list.append(R0)
        index_list.append(index)
    
    return index_list, R0_list


if __name__ == "__main__":

    model1 = SI_model(N=10**5, k=5, i=0.01, init_i=0.1, network="random")
    model2 = SI_model(N=10**5, k=0.8, i=0.1, init_i=0.1, network="random")

    t_steps = 200
    reps = 2

    plt.figure()

    model1_infected = ex_simulation(model1, reps, t_steps)
    model2_infected = ex_simulation(model2, reps, t_steps)

    print("i) The R0 for model 1 (N=10**5, k=5, i=0.01) at t=0 equals", (R0(model1_infected)[1][0]))
    print("ii) The R0 for model 2 (N=10**5, k=0.8, i=0.1) at t=0 equals", (R0(model2_infected)[1][0]))
    print("The ratio between case i) and case ii) equals", round((R0(model1_infected)[1][0])/(R0(model2_infected)[1][0]),2))

    plt.xlabel("t", fontsize=14)
    plt.ylabel(r"$\frac{I}{N}$", fontsize=16)
    plt.legend()
    plt.savefig("Ep_1b.png")

    plt.clf()
    index_list, R0_list = R0(model1_infected)
    plt.plot(index_list, R0_list)

    index_list, R0_list = R0(model2_infected)
    plt.plot(index_list, R0_list)
    plt.show()

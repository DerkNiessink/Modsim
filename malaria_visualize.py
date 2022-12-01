import numpy as np
import matplotlib

# matplotlib.use('TkAgg') # Mac specific
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, height, width, pauseTime=0.1):
        """
        This simple visualization shows the population of mosquitos and humans.
        Each subject is color coded according to its state.
        """
        self.h = height
        self.w = width
        self.pauseTime = pauseTime
        grid = np.zeros((self.w, self.h))
        self.im = plt.imshow(grid, vmin=-4, vmax=2, cmap="rainbow")
        """
        Color information
        """
        fig = plt.gcf()
        fig.text(0.02, 0.6, "Free space", color="palegoldenrod", fontsize=14)

        fig.text(0.02, 0.5, "M: inf", color="red", fontsize=14)
        fig.text(0.02, 0.45, "M: not-inf", color="darkorange", fontsize=14)

        fig.text(0.02, 0.35, "H: sus", color="aquamarine", fontsize=14)
        fig.text(0.02, 0.3, "H: inf", color="darkturquoise", fontsize=14)
        fig.text(0.02, 0.25, "H: imm", color="royalblue", fontsize=14)
        fig.text(0.02, 0.20, "H: Q", color="blueviolet", fontsize=14)
        plt.subplots_adjust(left=0.3)

    def update(self, t, mosquitoPopulation, humanPopulation):
        """
        Updates the data array, and draws the data.
        """
        grid = np.zeros((self.w, self.h))

        """
        Visualizes the infected vs non-infected mosquitos (2, 1) respectively.
        Visualizes the susceptible, infected and immune humans (-1, -2, -3)
        respectively.
        """
        for m in mosquitoPopulation:
            if m.infected:
                grid[m.position[0]][m.position[1]] = 2
            else:
                grid[m.position[0]][m.position[1]] = 1

        for h in humanPopulation:
            if h.state == "S":
                grid[h.position[0]][h.position[1]] = -1
            elif h.state == "I":
                grid[h.position[0]][h.position[1]] = -2
            elif h.state == "Q":
                grid[h.position[0]][h.position[1]] = -4
            else:
                grid[h.position[0]][h.position[1]] = -3

        self.im.set_data(grid)

        plt.draw()
        plt.title("t = %i" % t)
        plt.pause(0.01)

    def persist(self):
        """
        Use this method if you want to have the visualization persist after the
        calling the update method for the last time.
        """
        plt.show()


"""
* EXAMPLE USAGE *

sim = Model()
vis = visualization.Visualization(sim.height, sim.width)
maxT = 100
for t in range(maxT):
    sim.update()
    vis.update()
vis.persist()
"""

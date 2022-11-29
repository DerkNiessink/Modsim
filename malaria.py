import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import malaria_visualize


class Model:
    def __init__(
        self,
        width=50,
        height=50,
        nHuman=2000,
        nMosquito=400,
        initMosquitoHungry=0.5,
        initHumanInfected=0.001,
        humanInfectionProb=1,
        mosquitoInfectionProb=0.9,
        biteProb=1.0,
        hungryTime=5,
        HumanDieProb=0.05,
        dieTime=30,
    ):
        """
        Model parameters
        Initialize the model with the width and height parameters.
        """
        self.height = height
        self.width = width
        self.nHuman = nHuman
        self.nMosquito = nMosquito
        self.humanInfectionProb = humanInfectionProb
        self.mosquitoInfectionProb = mosquitoInfectionProb
        self.biteProb = biteProb
        self.hungryTime = hungryTime
        self.HumanDieProb = HumanDieProb
        self.dieTime = dieTime

        """
        Data parameters
        To record the evolution of the model
        """
        self.human_properties = []
        self.occupied_cells = []
        self.infectedCount = 0
        self.deathCount = 0
        # etc.

        """
        Population setters
        Make a data structure in this case a list with the humans and mosquitos.
        """
        self.humanPopulation = self.set_human_population(initHumanInfected)
        self.mosquitoPopulation = self.set_mosquito_population(initMosquitoHungry)

    def set_human_population(self, initHumanInfected):
        """
        This function makes the initial human population, by iteratively adding
        an object of the Human class to the humanPopulation list.
        The position of each Human object is randomized. A number of Human
        objects is initialized with the "infected" state.
        """
        humanPopulation = []
        for i in range(self.nHuman):
            x, y = self.get_free_position()
            self.occupied_cells.append((x, y))

            if (i / self.nHuman) <= initHumanInfected:
                state = "I"  # I for infected
                self.infectedCount += 1
            else:
                state = "S"  # S for susceptible
            humanPopulation.append(Human(x, y, state))
            self.human_properties.append((x, y, state, 0))

        return humanPopulation

    def get_free_position(self):
        x = np.random.randint(self.width)
        y = np.random.randint(self.height)

        while (x, y) in self.occupied_cells:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)

        return x, y

    def set_mosquito_population(self, initMosquitoHungry):
        """
        This function makes the initial mosquito population, by iteratively
        adding an object of the Mosquito class to the mosquitoPopulation list.
        The position of each Mosquito object is randomized.
        A number of Mosquito objects is initialized with the "hungry" state.
        """
        mosquitoPopulation = []
        for i in range(self.nMosquito):
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            if (i / self.nMosquito) <= initMosquitoHungry:
                hungry = True
            else:
                hungry = False
            mosquitoPopulation.append(Mosquito(x, y, hungry))
        return mosquitoPopulation

    def update(self):
        """
        Perform one timestep:
        1.  Update mosquito population. Move the mosquitos. If a mosquito is
            hungry it can bite a human with a probability biteProb.
            Update the hungry state of the mosquitos.
        2.  Update the human population. If a human dies remove it from the
            population, and add a replacement human.
        """

        # Update musquito population
        for i, m in enumerate(self.mosquitoPopulation):

            m.move(self.height, self.width)
            m.update_hungriness(self.hungryTime)

            for h in self.humanPopulation:
                if (
                    m.position == h.position
                    and m.hungry
                    and np.random.uniform() <= self.biteProb
                ):
                    m.bite(h, self.humanInfectionProb, self.mosquitoInfectionProb)

        # Update human population
        for j, h in enumerate(self.humanPopulation):

            h.update_sickness(self.dieTime)

            if h.state == "I" and np.random.uniform() <= self.HumanDieProb:
                # Remove dead  human
                self.humanPopulation.pop(j)
                self.occupied_cells.pop(j)

                # Add newborn human
                x, y = self.get_free_position()
                self.humanPopulation.append(Human(x, y, "S"))
                self.occupied_cells.append((x, y))

        # Update data/statistics
        """
        To implement: update the data/statistics e.g. infectedCount,
                      deathCount, etc.
        """
        return self.infectedCount, self.deathCount


class Mosquito:
    def __init__(self, x, y, hungry):
        """from collections import Counter
        Class to model the mosquitos. Each mosquito is initialized with a random
        position on the grid. Mosquitos can start out hungry or not hungry.
        All mosquitos are initialized infection free (this can be modified).
        """
        self.position = [x, y]
        self.hungry = hungry
        self.infected = False
        self.time_not_hungry = 0

    def bite(self, human, humanInfectionProb, mosquitoInfectionProb):
        """
        Function that handles the biting. If the mosquito is infected and the
        target human is susceptible, the human can be infected.
        If the mosquito is not infected and the target human is infected, the
        mosquito can be infected.
        After a mosquito bites it is no longer hungry.
        """
        if self.infected and human.state == "S":
            if np.random.uniform() <= humanInfectionProb:
                human.state = "I"
        elif not self.infected and human.state == "I":
            if np.random.uniform() <= mosquitoInfectionProb:
                self.infected = True
        self.hungry = False
        self.time_not_hungry = 0

    def update_hungriness(self, hungryTime):
        if self.hungry == False:
            self.time_not_hungry += 1
            if self.time_not_hungry > hungryTime:
                self.hungry = True

    def move(self, height, width):
        """
        Moves the mosquito one step in a random direction.
        """
        deltaX = np.random.randint(-1, 2)
        deltaY = np.random.randint(-1, 2)

        # % width and height for periodic boundary conditions
        self.position[0] = (self.position[0] + deltaX) % width
        self.position[1] = (self.position[1] + deltaY) % height


class Human:
    def __init__(self, x, y, state):
        """
        Class to model the humans. Each human is initialized with a random
        position on the grid. Humans can start out susceptible or infected
        (or immune).
        """
        self.position = [x, y]
        self.state = state
        self.time_sick = 0

    def update_sickness(self, sickTime):
        if self.state == "I":
            self.time_sick += 1
            if self.time_sick > sickTime:
                self.state = "Imm"


if __name__ == "__main__":
    """
    Simulation parameters
    """
    fileName = "simulation"
    timeSteps = 200
    t = 0
    plotData = False
    """
    Run a simulation for an indicated number of timesteps.
    """
    file = open(fileName + ".csv", "w")
    sim = Model()
    vis = malaria_visualize.Visualization(sim.height, sim.width)
    print("Starting simulation")
    while t < timeSteps:
        [d1, d2] = sim.update()  # Catch the data
        line = (
            str(t) + "," + str(d1) + "," + str(d2) + "\n"
        )  # Separate the data with commas
        file.write(line)  # Write the data to a .csv file
        vis.update(t, sim.mosquitoPopulation, sim.humanPopulation)
        t += 1
    file.close()
    vis.persist()

    if plotData:
        """
        Make a plot by from the stored simulation data.
        """
        data = np.loadtxt(fileName + ".csv", delimiter=",")
        time = data[:, 0]
        infectedCount = data[:, 1]
        deathCount = data[:, 2]
        plt.figure()
        plt.plot(time, infectedCount, label="infected")
        plt.plot(time, deathCount, label="deaths")
        plt.legend()
        plt.show()

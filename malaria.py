"""
"Malaria.py"
- Developed by Jenna de Vries and Derk Niessink

Model for simulating the infectious Malaria disease using a 2D cellular
automaton. The model allows for testing the effect of a quarantine prevention
method.

Usage:

- Model parameters can be adjusted in the __innit__ of the class Model

- Simulation parameters can be adjusted under the if __name__ == "__main__": statement.
    ->  Simulations will be run for no isolation and all the isolation values
        in the "isolation_times" list. Leave the list empty to only run for no
        isolation.
    ->  Set "plotting" is true to save the two plots as "deathcount.png" and
        "infectedcount.png"
    ->  Set "visualize" is true to show the live grid while the simulation is
        running (will impact performance negatively).

- Data of the infected count and death count of every simulation will be saved
as "simulation{i}.csv". The highest {i} corresponds to no isolation and the others
are in order of the "isolation_times" list.
"""


import matplotlib.pyplot as plt
import numpy as np
import malaria_visualize


class Model:
    def __init__(
        self,
        width=50,
        height=50,
        nHuman=400,
        nMosquito=2000,
        initMosquitoHungry=0.1,
        initHumanInfected=0.1,
        humanInfectionProb=0.9,
        mosquitoInfectionProb=0.9,
        biteProb=0.9,
        hungryTime=10,
        lifeSpan=49,
        dieNaturalCauses=1 / 1000,
        HumanDieProb=1 / 10,
        sickTime=35,
        isolation=False,
        isolationTime=10,
        immuneTime=90,
    ):
        """
        Model parameters
        """
        self.height = height
        self.width = width
        self.nHuman = nHuman
        self.nMosquito = nMosquito
        self.humanInfectionProb = humanInfectionProb
        self.mosquitoInfectionProb = mosquitoInfectionProb
        self.biteProb = biteProb
        self.hungryTime = hungryTime
        self.lifeSpan = lifeSpan
        self.HumanDieProb = HumanDieProb
        self.dieNaturalCauses = dieNaturalCauses
        self.sickTime = sickTime
        self.isolation = isolation
        self.isolationTime = isolationTime
        self.immuneTime = immuneTime

        """
        Data parameters
        To record the evolution of the model
        """
        self.occupied_cells = []
        self.infectedCount = 0
        self.deathCount = 0

        """
        Population setters
        Make a list with the humans and mosquitos.
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
            humanPopulation.append(
                Human(
                    x,
                    y,
                    state,
                    self.sickTime,
                    self.isolationTime,
                    self.isolation,
                    self.immuneTime,
                )
            )

        return humanPopulation

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
            mosquitoPopulation.append(
                Mosquito(
                    x,
                    y,
                    hungry,
                    self.lifeSpan,
                    self.mosquitoInfectionProb,
                    self.hungryTime,
                )
            )
        return mosquitoPopulation

    def get_free_position(self):
        """
        Get a free position in the grid. A free position is defined by a
        position that does not contain a human.
        """
        x = np.random.randint(self.width)
        y = np.random.randint(self.height)

        while (x, y) in self.occupied_cells:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)

        return x, y

    def update(self):
        """
        Perform one timestep:
        1.  Update mosquito population. Move the mosquitos. If a mosquito is
            hungry it can bite a human with a probability biteProb.
            Update the hungry state of the mosquitos.
        2.  Update the human population. If a human dies remove it from the
            population, and add a replacement human.
        """

        """
        Update musquito population
        """
        for i, m in enumerate(self.mosquitoPopulation):

            m.move(self.height, self.width)
            m.update_hungriness()
            m.update_infectioness()

            for h in self.humanPopulation:
                if (
                    m.position == h.position
                    and m.hungry
                    and np.random.uniform() <= self.biteProb
                ):
                    m.bite(h, self.humanInfectionProb)

        """"
        Update human population
        """
        self.infectedCount = 0
        for j, h in enumerate(self.humanPopulation):

            h.update_sickness()
            h.update_immunity()

            if np.random.uniform() <= self.dieNaturalCauses:
                self.replace_human(j)

            if h.state == "I" or h.state == "Q":
                self.infectedCount += 1

                if np.random.uniform() <= self.HumanDieProb:
                    self.replace_human(j)
                    self.deathCount += 1

        return self.infectedCount, self.deathCount

    def replace_human(self, index):
        """
        Replace a human in the population with a human in state susceptible
        and with a new position.
        """
        self.humanPopulation.pop(index)
        self.occupied_cells.pop(index)

        x, y = self.get_free_position()
        self.humanPopulation.append(
            Human(
                x,
                y,
                "S",
                self.sickTime,
                self.isolationTime,
                self.isolation,
                self.immuneTime,
            )
        )
        self.occupied_cells.append((x, y))


class Mosquito:
    def __init__(self, x, y, hungry, lifeSpan, mosquitoInfectionProb, hungryTime):
        """
        Class to model the mosquitos. Each mosquito is initialized with a random
        position on the grid. Mosquitos can start out hungry or not hungry.
        All mosquitos are initialized infection free (this can be modified).
        """
        self.position = [x, y]
        self.hungry = hungry
        self.infected = False
        self.time_not_hungry = 0
        self.lifeSpan = lifeSpan
        self.InfectionProb = mosquitoInfectionProb
        self.hungryTime = hungryTime

    def bite(self, human, humanInfectionProb):
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
            if np.random.uniform() <= self.InfectionProb:
                self.infected = True
        self.hungry = False
        self.time_not_hungry = 0

    def update_hungriness(self):
        """
        If the mosquito is not hungry, update the time that the mosquito has
        been not hungry and let it be hungry after the given hungry time.
        """
        if not self.hungry:
            self.time_not_hungry += 1
            if self.time_not_hungry > self.hungryTime:
                self.hungry = True

    def update_infectioness(self):
        """
        Change the mosquito from infected to not infected with a probability
        dependent on the life time of a mosquito.
        """
        chanceToDie = 1 / self.lifeSpan
        rand = np.random.rand()
        if rand < chanceToDie:
            self.infected = False

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
    def __init__(self, x, y, state, sickTime, isolationTime, isolation, immuneTime):
        """
        Class to model the humans. Each human is initialized with a random
        position on the grid. Humans can start out susceptible or infected.
        """
        self.position = [x, y]
        self.state = state
        self.time_sick = 0
        self.time_immune = 0
        self.sickTime = sickTime
        self.isolationTime = isolationTime
        self.isolation = isolation
        self.immuneTime = immuneTime

    def update_sickness(self):
        """
        If human is sick, update the time that it has been sick. Set human in
        quarantine after the given time and set human to immune if it has
        survived the disease.
        """
        if self.state == "I" or self.state == "Q":
            self.time_sick += 1

            if self.time_sick > self.isolationTime and self.isolation == True:
                self.state = "Q"  # Q for quarantine

            if self.time_sick > self.sickTime:
                self.state = "Imm"  # Imm for immune
                self.time_sick == 0

    def update_immunity(self):
        """
        If the human is immune, update the time that it has been immune and
        change its state to suscepible after the given immune time.
        """
        if self.state == "Imm":
            self.time_immune += 1

            if self.time_immune > self.immuneTime:
                self.state = "S"
                self.time_immune = 0


if __name__ == "__main__":
    """
    Simulation parameters
    """
    isolation_times = [2, 5, 9, 19]
    plotData = True
    visualize = False
    timeSteps = 2000

    """
    Run simulations without isolation and for the given isolation times
    """
    for i in range(len(isolation_times) + 1):
        fileName = f"simulation{i}"
        t = 0
        file = open(fileName + ".csv", "w")
        if i == len(isolation_times):
            sim = Model(isolation=False)
        else:
            sim = Model(isolation=True, isolationTime=isolation_times[i])

        if visualize:
            vis = malaria_visualize.Visualization(sim.height, sim.width)
        print(f"Running simulation {i}")

        """
        Run the simulation for the given timesteps
        """
        while t < timeSteps:
            [d1, d2] = sim.update()  # Catch the data
            line = str(t) + "," + str(d1) + "," + str(d2) + "\n"
            file.write(line)

            if visualize:
                vis.update(t, sim.mosquitoPopulation, sim.humanPopulation)
            t += 1
        file.close()
        if visualize:
            vis.persist()

    if plotData:
        """
        Make a plot from the stored simulation data.
        """

        data_no_isolation = np.loadtxt(
            f"simulation{len(isolation_times)}.csv", delimiter=","
        )
        time = data_no_isolation[:, 0]
        infectedCount = data_no_isolation[:, 1]
        deathCount = data_no_isolation[:, 2]

        plt.figure()

        # Plot deathcount without isolation
        plt.plot(time, deathCount, label="No isolation")

        # Plot data with several values of isolation times
        for i, isolation_time in enumerate(isolation_times):
            data_isolation = np.loadtxt(f"simulation{i}.csv", delimiter=",")
            deathcount_isolation = data_isolation[:, 2]
            plt.plot(
                time,
                deathcount_isolation,
                label=f"{isolation_time} days until isolation",
            )

        plt.xlabel("Time in days")
        plt.ylabel("Number of human deaths")
        plt.legend()
        plt.savefig("deathcount.png")
        plt.show()

        plt.figure()
        # Plot infected count without isolation
        percentageInfectedCount = [_ / sim.nHuman for _ in data_no_isolation[:, 1]]
        plt.plot(time, percentageInfectedCount, label="No isolation")

        # Plot infected count with several values of isolation times
        for i, isolation_time in enumerate(isolation_times):
            data_isolation = np.loadtxt(f"simulation{i}.csv", delimiter=",")
            infectedCount_isolation = data_isolation[:, 1]
            percentageInfectedCount_iso = [_ / sim.nHuman for _ in data_isolation[:, 1]]
            plt.plot(
                time,
                percentageInfectedCount_iso,
                label=f"{isolation_time} days until isolation",
            )

        plt.xlabel("Time in days")
        plt.ylabel("Percentage of humans infected")
        plt.legend()
        plt.savefig("infectedcount.png")
        plt.show()

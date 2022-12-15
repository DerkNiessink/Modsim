from epidemics1 import SI_model
from epidemics1 import ex_simulation

import matplotlib.pyplot as plt


model1 = SI_model(N=10**5, k=5, i=0.01, init_i=0.1, network="random")
model2 = SI_model(N=10**5, k=5, i=0.01, init_i=0.1, network="scale_free", power=2.5)

t_steps = 100
reps = 1

plt.figure()
ex_simulation(model1, reps, t_steps)
ex_simulation(model2, reps, t_steps)
plt.xlabel("t", fontsize=14)
plt.ylabel(r"$\frac{I}{N}$", fontsize=16)
plt.legend(labels=["random", "scale_free"])
plt.savefig("Ep_4a.png")

plt.figure()
plt.plot(model1.time, model1.avg_k_infections)
plt.plot(model2.time, model2.avg_k_infections)
plt.xlabel("t", fontsize=14)
plt.ylabel(r"$\langle k \rangle$ of new infections", fontsize=14)
plt.legend(labels=["random", "scale_free"])
plt.savefig("Ep_4c.png")

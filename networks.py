import matplotlib.pyplot as plt

"""
1a (questions)
"""
degrees = [1, 2, 3, 2, 2]

plt.ylabel("Frequency")
plt.xlabel("Degrees")
plt.hist(degrees, bins=3)
plt.savefig("figures/1a.png")

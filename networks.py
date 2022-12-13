import matplotlib.pyplot as plt
import numpy as np
"""
1a (questions)
"""
degrees = [1,2,3,2,2]

plt.ylabel("Frequency")
plt.xlabel("Degrees")
plt.hist(degrees, bins = 3)
plt.savefig("figures/1a.png")

"""
1b 
"""
import networkx as nx
def b1(N, k, time):
    G = nx.fast_gnp_random_graph(n=N, p=k/N)
    nx.set_node_attributes(G, 0)

    # Initial condition, 10% infected
    plt.clf()
    nx.draw(G)
    plt.show()

    for t in range (0,time):
        

    # Check mean amount of neighbors 
    count_list=[]
    for x in range(0,k):
        neighbors = nx.all_neighbors(G, x)
        count=0
        for _ in neighbors:
            count+=1
        count_list.append(count)

    return np.mean(count_list)
print(b1(100, 5))
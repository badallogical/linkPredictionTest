import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

from methods import SULP

G = nx.read_edgelist('./fb-pages-food.edges')

print("nodes : " , len(G.nodes()))
print("edges : " ,len(G.edges()))

N = [int] * len(G.nodes())
nodes = G.nodes()
map = {}

# map DOID with node index from 0 to nodes.
k = 0;
for i in nodes:
    map[i] = k
    k = k + 1

# get the neighbours for each node i
for i in nodes:
    N[map[i]] = np.array([n for n in G.neighbors(i)])

# find all non edges ( possible future links )
non_edges = [i  for i in nx.non_edges(G)]
print("non edges ( possible links ) " , len(non_edges))

# find similarity value p(k) for possible links
map2 = {}
for edge in non_edges:
    similarity = SULP(N,map[edge[0]], map[edge[1]])
    # similarity = len( np.intersect1d( N[map[edge[0]]] , N[map[edge[1]]] )) / ( len(N[map[edge[0]]]) * math.sqrt(len(N[map[edge[1]]])))
    print(edge, map[edge[0]],map[edge[1]], " p(k) = ", similarity )
    map2[edge] = similarity


fig = plt.figure(1, figsize=(20, 20), dpi=100)
nx.draw(G,with_labals=True,node_size = 40, alpha = 0.6, width = 1)
plt.show()
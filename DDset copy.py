import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

from methods import SULP

G = nx.read_edgelist('./fb-pages-food.edges')
edges = G.edges()

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

#set color of edges 
for edge in edges:
  G[edge[0]][edge[1]]['color'] = 'blue';
  G[edge[0]][edge[1]]['width'] = 1;

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
    G.add_edge(map[edge[0]], map[edge[1]]);
    G[map[edge[0]]][map[edge[1]]]['color'] = 'red'
    G[map[edge[0]]][map[edge[1]]]['width'] = 0 if(similarity < 1 ) else 2*similarity


colors = [ G[u][v]['color'] for u,v in G.edges() ]
weight = [ G[u][v]['width'] for u,v in G.edges() ]
fig = plt.figure(1, figsize=(20, 20), dpi=100)
# nx.draw(G,edge_color=colors, width=weight,node_size = 40, alpha = 0.6)
nx.draw(G,pos = nx.random_layout(G), edge_color=colors,width=weight,with_labels=True )
plt.show()
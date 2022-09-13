import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np


with open("./fb-pages-food.edges",encoding="utf8") as f:
    fb_edges = f.read().splitlines() 

with open("./fb-pages-food.nodes",encoding="utf8") as f:
    fb_nodes = f.read().splitlines() 


G = nx.Graph()
for i in range(0, len(fb_nodes)):
    G.add_node(i) 

for i in tqdm(fb_edges):
    edge = i.split(',')
    G.add_edge(edge[0], edge[1])


fig = plt.figure(1, figsize=(10, 10), dpi=100)

print(list(G.nodes()))
pos = nx.spring_layout(G)
nx.draw(G,pos = pos,node_size = 40, alpha = 0.6, width = 1)
plt.show()
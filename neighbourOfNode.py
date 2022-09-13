from cmath import log
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

from methods import SULP

def commmonNeighbour(N, i, j ):
  similarity = len ( np.intersect1d( N[i], N[j] ) );
  return similarity;

def jaccardCoefficient(N,i,j):
  similarity = len ( np.intersect1d( N[i], N[j] ) ) / ( len( np.union1d(N[i], N[j])));
  return similarity;

def adamicAdarIndex(N,i,j):
  mutualResources = np.intersect1d(N[i], N[j] )
  similarity = 0;
  for z in mutualResources:
    kz = len(N[z] );  # degree of mutual resourse z
    similarity += ( 1 / math.log(kz) );
  
  return similarity

def adamicAdarIndex(N,i,j):
  mutualResources = np.intersect1d(N[i], N[j] )
  similarity = 0;
  for z in mutualResources:
    kz = len(N[z] );  # degree of mutual resourse z
    similarity += ( 1 / math.log(kz) )
  
  return similarity

def resourceAllocationIndex(N, i, j ):
  mutualResources = np.intersect1d(N[i], N[j] )
  similarity = 0;
  for z in mutualResources:
    kz = len(N[z] );  # degree of mutual resourse z
    similarity += 1 / kz;   
  
  return similarity

  



# find the neighbours of each node.
G = nx.Graph()

for i in range(1,6):
  G.add_node(i);

G.add_edge(1,2)
G.add_edge(1,3)
G.add_edge(1,4)
G.add_edge(2,3)
G.add_edge(2,5)
G.add_edge(3,4)

nx.draw(G,with_labels=True)
N = [None] * (len(G.nodes()) +1)
nodes = G.nodes()
print("edges : ", len(G.edges()))



# get neighbours of  node i
for i in nodes:
  N[i] = np.array([n for n in G.neighbors(i)])

for i in nodes:
  print("neighbour %d => %s" % (i,N[i]) , "length ", len(N[i]) )

# find common neighbour between , Ni intersection Nj ( similarity function(x,y) )
for i in nodes:
  for j in nodes:
    if( i != j ):
      print( "N[{}] , N[{}] = > ".format(i,j) ,np.intersect1d(N[i], N[j]))


# find all the unconnected nodes p(K) i.e similarity value
adj = nx.to_numpy_array(G)
print(adj)

non_edges = [ i for i in nx.non_edges(G)]
for k in non_edges:
  similarity = SULP(N,k[0],k[1])
  print( k , N[k[0]], N[k[1]] , " intersection " ,np.intersect1d(N[k[0]], N[k[1]]) , " p(k) = ", similarity);


plt.show()
      
  

  
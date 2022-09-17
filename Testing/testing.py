from random import random
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from methods import SULP
import numpy as np

fb_df = pd.read_csv('./Testing/fb-pages-food.edges')

G = nx.from_pandas_edgelist(fb_df, 'node_1', 'node_2',None,create_using= nx.Graph())

# spliting train , test data fro non edges 
edge_train, edge_test = train_test_split(fb_df,test_size=0.3,random_state=32)

print("data " ,fb_df.size)
print("test data " ,len(edge_test))
print("train data " ,len(edge_train))

test_graph = nx.from_pandas_edgelist(edge_test, 'node_1', 'node_2',None, nx.Graph())

print("graph edge " ,len(test_graph.edges()))
print("test edges : ", edge_test)

plt.figure(1,dpi=100)
pos= nx.random_layout(G)
nx.draw(G, pos= pos,node_color="black", node_size = 40, alpha = 0.6)
plt.savefig('./Testing/g_before.png')

# find prediction for actual  edges 
removed_edge = []
for index,row in edge_test.iterrows():
    G.remove_edge(row['node_1'], row['node_2'])
    removed_edge.append([row['node_1'], row['node_2']])

plt.figure(2);
nx.draw(G, pos= pos,node_color="black", node_size = 40, alpha = 0.6)
plt.savefig('./Testing/g_after.png')

for edge in G.edges:
  G[edge[0]][edge[1]]['color'] = 'green'
  G[edge[0]][edge[1]]['width'] = 1

TP = 0; FP = 0; TN = 0; FN = 0;
for edge in removed_edge:
    print( edge )
    ni = np.array([n for n in G.neighbors(edge[0])])
    nj = np.array([n for n in G.neighbors(edge[1])])
    similarity = SULP(ni,nj)
    print(edge , "similarity : ", similarity )
    if( similarity >= 0.6 ):
        G.add_edge(edge[0], edge[1])
        G[edge[0]][edge[1]]['color']= "red"
        G[edge[0]][edge[1]]['width'] = 2
        TP = TP + 1
    else:
        FN = FN + 1

# print("Prediction correct : " , (count / len(removed_edge)) * 100 )
# print("Correct Prediction : " , count, "Total Test Edges : ", len(removed_edge) )



# find prediction for actual non edges f
non_edges_df = pd.DataFrame([ i for i in nx.non_edges(G) ], columns=["node_1", "node_2"]) 

# spliting train , test data fro non edges 
non_edge_train, non_edge_test = train_test_split( non_edges_df, test_size=0.3, random_state=32)

print("Total non_edges :", len(non_edges_df) )
print("Non edges for test : ", len(non_edge_test))
print("Non edges for train : ", len(non_edge_train))

actual_non_edges_test = []
for index, row in non_edge_test.iterrows():
  actual_non_edges_test.append([row['node_1'], row['node_2']])

for non_edge in actual_non_edges_test:
  ni = np.array([n for n in G.neighbors(non_edge[0])])
  nj = np.array([n for n in G.neighbors(non_edge[1])])
  similarity = SULP(ni,nj)
  if( similarity >= 0.6 ):
        G.add_edge(non_edge[0], non_edge[1])
        G[non_edge[0]][non_edge[1]]['color']= "blue"
        G[non_edge[0]][non_edge[1]]['width'] = 2
        FP = FP + 1
  else:
      TN = TN + 1

# Precission
precision = TP / ( TP + FP )

#recall
recall = TP / ( TP + FN )

#accuracy
accuracy = (TP + TN) / ( TP + TN + FP + FN )

print("Total Test edges : " , (TP + TN + FP + FN ) , "or ", len(edge_test)+len(non_edge_test))
print("Precision: ", precision)
print("Recall : ", recall)
print("Accuracy : ", accuracy)

colors = [G[u][v]['color'] for u,v in G.edges()]
widths = [G[u][v]['width'] for u,v in G.edges()]
plt.figure(3)
nx.draw(G,pos=pos,edge_color=colors, node_color="black", width=widths, node_size = 40, alpha = 0.6)
plt.savefig('./Testing/g_after_prediction.png')
plt.show()


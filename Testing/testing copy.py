from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from methods import SULP
import numpy as np

fb_df = pd.read_csv('./Testing/test.csv')

G = nx.from_pandas_edgelist(fb_df, 'node_1', 'node_2',None,create_using= nx.Graph())

edge_train, edge_test = train_test_split(fb_df,test_size=0.3,random_state=32)

print("data " ,fb_df.size)
print("test data " ,len(edge_test))
print("train data " ,len(edge_train))

g2 = nx.from_pandas_edgelist(edge_test, 'node_1', 'node_2',None, nx.Graph())

print("graph edge " ,len(g2.edges()))
print("test edges : ", edge_test)

plt.figure(1,dpi=100);
pos=nx.random_layout(G)
nx.draw(G, pos= pos, with_labels=True);
plt.savefig('./Testing/g_before.png');

remove_edge = []
for index,row in edge_test.iterrows():
    G.remove_edge(row['node_1'], row['node_2'])
    remove_edge.append([row['node_1'], row['node_2']])

plt.figure(2);
nx.draw(G, pos= pos,with_labels=True);
plt.savefig('./Testing/g_after.png')

for edge in G.edges:
  G[edge[0]][edge[1]]['color'] = 'blue';
  G[edge[0]][edge[1]]['width'] = 1;

for edge in remove_edge:
    print( edge)
    ni = np.array([n for n in G.neighbors(edge[0])])
    nj = np.array([n for n in G.neighbors(edge[1])])
    similarity = SULP(ni,nj)
    print(edge , "similarity : ", similarity )
    if( similarity >= 0.9 ):
        G.add_edge(edge[0], edge[1])
        G[edge[0]][edge[1]]['color']= "red"
        G[edge[0]][edge[1]]['width'] = 2

colors = [G[u][v]['color'] for u,v in G.edges()]
widths = [G[u][v]['width'] for u,v in G.edges()]
plt.figure(3)
nx.draw(G,pos=pos,edge_color=colors, width=widths, with_labels=True);
plt.savefig('./Testing/g_after_prediction.png')



plt.show()






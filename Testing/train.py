from random import random
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from methods import SULP
import numpy as np
from sklearn import svm

dd_df = pd.read_csv('./Testing/DD-Miner_miner-disease-disease.tsv')

G = nx.from_pandas_edgelist(dd_df, 'DOID1', 'DOID2',None,create_using= nx.Graph())

# spliting train , test data from actual edges 
train_graph, test_graph = train_test_split(dd_df,test_size=0.3,random_state=32)

#based on edge train only, and removing the edge test data set assuming they will be connected later ( actually ).
#also removed edge must no be an isolated node.

#find all the non-edges from the training data of graph.
non_edges_training_graph = pd.DataFrame([ i for i in nx.non_edges(G) ], columns=["DOID1", "DOID2"])
non_edges_training_graph['link'] = 0

print(non_edges_training_graph.head())


#prepare the training data set with the link prediction of SULP
for index, row in non_edges_training_graph.iterrows():
    # get the neigbors for each node of non edge ni and nj 
    ni = np.array([ n for n in G.neighbors(row['DOID1']) ])
    nj = np.array([ n for n in G.neighbors(row['DOID2']) ])

    # get similarity value and update link of each non edge
    similarity = SULP(ni, nj)
    if( similarity > 0.6 ):
        non_edges_training_graph.at[index,'link'] = 1


#feature and target extraction
target_link = non_edges_training_graph['link']
feature_edge = non_edges_training_graph.drop('link',axis=1)
print("target link "  , target_link.head())
print("feature edge ", feature_edge.head()) 

# train model
clf = svm.SVC()
clf.fit(feature_edge, target_link)
clf()





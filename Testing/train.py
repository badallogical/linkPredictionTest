from random import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from methods import SULP
import numpy as np
from sklearn import svm

dd_df = pd.read_csv('./Testing/DD-Miner_miner-disease-disease.tsv')

G = nx.from_pandas_edgelist(dd_df, 'DOID1', 'DOID2',None,create_using= nx.Graph())

# spliting train , test data from actual edges 
train_graph_df, test_graph_df = train_test_split(dd_df,test_size=0.3,random_state=32)

# based on edge train only, and removing the edge test data set assuming they will be connected later ( actually ).
# also removed edge must no be an isolated node.

# find all the non-edges from the training data of graph.
training_graph = nx.from_pandas_edgelist(train_graph_df,"DOID1", "DOID2", None, create_using=nx.Graph())
non_edges_training_graph_df = pd.DataFrame([ i for i in nx.non_edges(training_graph) ], columns=["DOID1", "DOID2"])
non_edges_training_graph_df['link'] = 0
print("total edges", train_graph_df.size )
print("non edges", non_edges_training_graph_df.size )
print(non_edges_training_graph_df.head())

# prepare the training data set with the link prediction of SULP
for index, row in non_edges_training_graph_df.iterrows():
    # get the neigbors for each node of non edge ni and nj 
    ni = np.array([ n for n in G.neighbors(row['DOID1']) ])
    nj = np.array([ n for n in G.neighbors(row['DOID2']) ])

    # get similarity value and update link of each non edge
    similarity = SULP(ni, nj)
    if( similarity > 0.6 ):
        non_edges_training_graph_df.at[index,'link'] = 1


# feature and target extraction
training_target_link = non_edges_training_graph_df['link']
training_feature_edge = non_edges_training_graph_df.drop('link',axis=1)
print("target link "  , training_target_link.head())
print("feature edge ", training_feature_edge.head()) 

# train model
clf = svm.SVC()
clf.fit(training_feature_edge, training_target_link)
clf()

# prepare testing data set
testing_graph = nx.from_pandas_edgelist(test_graph_df,"DOID1", "DOID2",None, nx.Graph())
non_edges_testing_df = pd.DataFrame( [ n for n in nx.non_edges(testing_graph)], columns=["DOID1","DOID2"])
edges_testing_df = pd.DataFrame( [n for n in nx.edges(testing_graph)], columns=["DOID1","DOID2"] )
non_edges_testing_df['link'] = 0
edges_testing_df['link'] = 1
testing_graph_ready_df = pd.concat([non_edges_testing_df,edges_testing_df])
print("total edges" , test_graph_df.shape)
print("total edges", edges_testing_df.shape)
print("total non_edges", non_edges_testing_df.shape)
print("total size of testing ", testing_graph_ready_df.shape)
print("testing_data haed\n", testing_graph_ready_df.head())
print("testing_data_tail\n",testing_graph_ready_df.tail())

# feature and target extraction for testing dateset
testing_target_df = testing_graph_ready_df['link']
testing_feature_df = testing_graph_ready_df.drop('link', axis=1);


# model testing
training_model_pred = clf.predict(training_feature_edge)
testing_model_pred = clf.predict(testing_feature_df)

TP = 0; FP = 0; TN = 0; FN = 0;
for index, row in testing_target_df.iterrows():
    actual_link = row['link']
    pred_link = testing_model_pred.at[index,'link']

    if( actual_link == 1 and pred_link == 1 ):
        TP = TP + 1;
    elif( actual_link == 1 and pred_link == 0 ):
        TN = TN + 1;
    elif( actual_link == 0 and pred_link == 1 ):
        FP = FP + 1;
    else:
        FN = FN + 1     # if both are 0

# Precission
precision = TP / ( TP + FP )

#Recall
recall = TP / ( TP + FN )

#Accuracy
accuracy = (TP + TN) / ( TP + TN + FP + FN )

print("Total Test edges : " , (TP + TN + FP + FN ) , "or ", len(pred_link))
print("Precision: ", precision)
print("Recall : ", recall)
print("Accuracy : ", accuracy)
print("Accuracy: ", accuracy_score(testing_target_df, testing_model_pred))





import numpy as np
import math

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


def SULP(N,i,j):
    similarity = len(np.intersect1d(N[i], N[j]) ) / ( len(N[i]) * math.sqrt(len(N[j])) )
    return similarity
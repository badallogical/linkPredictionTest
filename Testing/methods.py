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


def SULP(ni,nj):
    if( len(ni) == 0 or len(nj) == 0 ):
      return 0
      
    similarity = len(np.intersect1d(ni, nj) ) / ( len(ni) * math.sqrt(len(nj)) )
    return similarity
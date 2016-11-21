import numpy as np

def nnrandInitializeWeights(L_in,L_out):
   #### randomly initialize weights for a given network connection in a neural network 
   # recommendation: take initialization parameter eps = sqrt(6)/sqrt(L_in + L_out) where
   # L_in is the number of incoming connections and L_out is the number of outgoing connections
   epsilon = np.sqrt(6)/np.sqrt(L_in + L_out)
   W = np.random.rand(L_out,L_in + 1) * 2 * epsilon - epsilon
   return W

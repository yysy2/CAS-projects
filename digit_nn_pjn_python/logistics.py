import numpy as np

def sigmoid(z):
   # computes the sigmoid function
   # alternative use from scipy: from scipy.stats import logistic - logistic.cdf(z)
   g = 1/(1+np.exp(-z))
   return g

def sigmoidGradient(z):
   # computes the gradient of the sigmoid function
   # g = zeros(np.shape(z))
   sz = sigmoid(z)
   g = sz*(1 - sz)
   return g

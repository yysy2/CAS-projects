import numpy as np
import costfunction as cf
def computeNumericalGradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
   # computes the gradient numerically by using finite differences
   diff = 1e-4
   perturb = np.zeros(theta.shape)
   numgrad = np.zeros(theta.shape)
   for i in range(0,theta.size):
      perturb[i] = diff
      (loss1, grad1) = cf.nnCostFunction(theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
      (loss2, grad2) = cf.nnCostFunction(theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
      numgrad[i] = (loss2 - loss1)/(2*diff)
      perturb[i] = 0
   return numgrad

def debugInitializeWeights(L_out, L_in):
   ### intializes the weights of a layer with L_in incoming connections and L_out outgoing connections
   W = np.zeros((L_out, L_in + 1))
   # initialize W using sin distribution
   blub = np.sin(np.arange(0,W.size))
   W = (blub.reshape(W.shape))/10
   return W

def checkNNGradients(lamb):
   #creates a small neural network to check the backpropagation gradients
   # it outputs the analytical gradient produced by the backpropagation code and the numerical 
   # gradients computed by computeNumericalGradient. Aim for similar values to validate
   input_layer_size = 4
   hidden_layer_size = 6
   num_labels = 4
   m = 5
   # generate random test data
   Theta1 = debugInitializeWeights(hidden_layer_size,input_layer_size)
   Theta2 = debugInitializeWeights(num_labels,hidden_layer_size)
   X = debugInitializeWeights(m,input_layer_size - 1)
   bla = np.mod(np.arange(0,m),num_labels)
   y = 1 + bla.transpose()
   # unroll parameters
   nn_params = np.hstack((Theta1.flatten(),Theta2.flatten()))
   (cost, grad) = cf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
   numgrad = computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
   # print grad,'grad'
   # print numgrad,'numgrad'
   fmt = '{:<8}{:<20}{}'
   print 'The following columns should be very similar:\n left the analytical, right the numerical gradient'
   print(fmt.format('', 'Analytical', 'Numerical'))
   for i, (gradi, numgradi) in enumerate(zip(grad, numgrad)):
      print(fmt.format(i, gradi, numgradi))
   diff_test = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)
   print diff_test,' this is the norm difference between the numerical and analytical solution - should be a very small value'
   return

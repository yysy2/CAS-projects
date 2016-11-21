import numpy as np
import logistics as lg
#from sklearn.utils.extmath import fast_dot

def nnCostFunction_vectorized_twoh(nn_params, input_layer_size, hidden_layer_1_size, hidden_layer_2_size, num_labels, X, y, lamb):
   # cost function for a two-layer neural network (input, hidden, output)
   # nn_params is a vector of unrolled parameters for the neural network-> to be converted back into weight matrices
   # return parameters: grad is the unrolled vector of the partial derivatives of the neural network

   #first step: reshape the nn_params vector back into weight matrices Theta1 and Theta2 for the two layers
   m_idx = hidden_layer_1_size*(input_layer_size+1)
   m_idx_2 = hidden_layer_2_size*(hidden_layer_1_size+1)
   Theta1 = nn_params[0:m_idx].reshape((hidden_layer_1_size, input_layer_size + 1))
   Theta2 = nn_params[m_idx:(m_idx+m_idx_2)].reshape((hidden_layer_2_size, hidden_layer_1_size + 1))
   Theta3 = nn_params[(m_idx+m_idx_2):].reshape((num_labels, hidden_layer_2_size + 1))
   m = len(X[:,0])
   J=0
   Theta1_grad = np.zeros(Theta1.shape)
   Theta2_grad = np.zeros(Theta2.shape)
   Theta3_grad = np.zeros(Theta3.shape)
   # start forward propagation 
   # print Theta1[0:3,0:3]
   # print Theta2[0:3,0:3]
   # print X[0:3,0:3]
   # exit()
   X = np.hstack((np.ones((m,1)),X))
   z2 = Theta1.dot(X.T)   
   a2 = lg.sigmoid(z2)
   a2 = np.hstack((np.ones((m,1)),a2.T))
   z3 = Theta2.dot(a2.T)
   a3 = lg.sigmoid(z3)
   a3 = np.hstack((np.ones((m,1)),a3.T))
   z4 = Theta3.dot(a3.T)
   htheta = lg.sigmoid(z4)
   htheta = htheta.T
   # start backpropagation: need approximate gradient for the neural network cost function
   y_matrix = []
   eye_matrix = np.eye(num_labels)
   for i in range(len(y)):
       y_matrix.append(eye_matrix[int(y[i]),:])
   y_matrix = np.array(y_matrix)
   J = np.sum(-np.multiply(y_matrix,np.log(htheta)) - np.multiply((1-y_matrix),np.log(1-htheta)),axis=None)
   J = J + lamb*(np.sum(Theta1[:,1:]**2,axis=None, dtype=np.float64) + np.sum(Theta2[:,1:]**2,axis=None, dtype=np.float64) + np.sum(Theta3[:,1:]**2,axis=None, dtype=np.float64))/2.0
   J = J/float(m)
   delta4 = htheta - y_matrix
   delta3 = (delta4.dot(Theta3[:,1:]))*lg.sigmoidGradient(z3[:,:].T)
   delta2 = (delta3.dot(Theta2[:,1:]))*lg.sigmoidGradient(z2[:,:].T)
   Theta3_grad = ((a3.T).dot(delta4)).T
   Theta2_grad = ((a2.T).dot(delta3)).T
   Theta1_grad = ((X.T).dot(delta2)).T
   Theta1_grad[:,1:] = Theta1_grad[:,1:] + lamb*Theta1[:,1:]
   Theta2_grad[:,1:] = Theta2_grad[:,1:] + lamb*Theta2[:,1:]
   Theta3_grad[:,1:] = Theta3_grad[:,1:] + lamb*Theta3[:,1:]
   Theta1_grad = Theta1_grad/float(m)
   Theta2_grad = Theta2_grad/float(m)
   Theta3_grad = Theta3_grad/float(m)
   # unroll gradients
   grad = np.hstack((Theta1_grad.flatten(),Theta2_grad.flatten(),Theta3_grad.flatten()))
   return (J, grad)

def nnCostFunction_vectorized(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
   # cost function for a two-layer neural network (input, hidden, output)
   # nn_params is a vector of unrolled parameters for the neural network-> to be converted back into weight matrices
   # return parameters: grad is the unrolled vector of the partial derivatives of the neural network

   #first step: reshape the nn_params vector back into weight matrices Theta1 and Theta2 for the two layers
   Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape((hidden_layer_size, input_layer_size + 1))
   Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape((num_labels, hidden_layer_size + 1))
   m = len(X[:,0])
   J=0
   Theta1_grad = np.zeros(Theta1.shape)
   Theta2_grad = np.zeros(Theta2.shape)
   # start forward propagation 
   # print Theta1[0:3,0:3]
   # print Theta2[0:3,0:3]
   # print X[0:3,0:3]
   # exit()
   X = np.hstack((np.ones((m,1)),X))
   z2 = Theta1.dot(X.T)   
   a2 = lg.sigmoid(z2)
   a2 = np.hstack((np.ones((m,1)),a2.T))
   z3 = Theta2.dot(a2.T)
   htheta = lg.sigmoid(z3)
   htheta = htheta.T
   # start backpropagation: need approximate gradient for the neural network cost function
   y_matrix = []
   eye_matrix = np.eye(num_labels)
   for i in range(len(y)):
       y_matrix.append(eye_matrix[int(y[i]),:])
   y_matrix = np.array(y_matrix)
   J = np.sum(-np.multiply(y_matrix,np.log(htheta)) - np.multiply((1-y_matrix),np.log(1-htheta)),axis=None)
   J = J + lamb*(np.sum(Theta1[:,1:]**2,axis=None, dtype=np.float64) + np.sum(Theta2[:,1:]**2,axis=None, dtype=np.float64))/2.0
   J = J/float(m)
   delta3 = htheta - y_matrix
   delta2 = (delta3.dot(Theta2[:,1:]))*lg.sigmoidGradient(z2[:,:].T)
   Theta2_grad = ((a2.T).dot(delta3)).T
   Theta1_grad = ((X.T).dot(delta2)).T
   Theta1_grad[:,1:] = Theta1_grad[:,1:] + lamb*Theta1[:,1:]
   Theta2_grad[:,1:] = Theta2_grad[:,1:] + lamb*Theta2[:,1:]
   Theta1_grad = Theta1_grad/float(m)
   Theta2_grad = Theta2_grad/float(m)
   # unroll gradients
   grad = np.hstack((Theta1_grad.flatten(),Theta2_grad.flatten()))
   return (J, grad)

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
   # cost function for a two-layer neural network (input, hidden, output)
   # nn_params is a vector of unrolled parameters for the neural network-> to be converted back into weight matrices
   # return parameters: grad is the unrolled vector of the partial derivatives of the neural network

   #first step: reshape the nn_params vector back into weight matrices Theta1 and Theta2 for the two layers
   Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape((hidden_layer_size, input_layer_size + 1))
   Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape((num_labels, hidden_layer_size + 1))
   m = len(X[:,0])
   J=0
   Theta1_grad = np.zeros(Theta1.shape)
   Theta2_grad = np.zeros(Theta2.shape)
   # start forward propagation 
   # print Theta1[0:3,0:3]
   # print Theta2[0:3,0:3]
   # print X[0:3,0:3]
   # exit()
   X = np.hstack((np.ones((m,1)),X))
   z2 = Theta1.dot(X.T)   
   a2 = lg.sigmoid(z2)
   a2 = np.hstack((np.ones((m,1)),a2.T))
   z3 = Theta2.dot(a2.T)
   htheta = lg.sigmoid(z3)
   htheta = htheta.T
   # start backpropagation: need approximate gradient for the neural network cost function
   delta3 = np.zeros((num_labels,1))
   delta2 = np.zeros((hidden_layer_size,1))
   for i in range(0,m):
      for k in range(0,num_labels):
         state_check = (int(y[i]) == k)
         J = J + (-state_check*np.log(htheta[i,k]) - (1 - state_check)*np.log(1 - htheta[i,k]))
         delta3[k] = htheta[i,k] - state_check
      Theta2_grad[:,:] = Theta2_grad[:,:] + delta3.dot(a2[i:i+1,:])
      delta2[:] = np.transpose(Theta2[:,1:]).dot(delta3)
      delta2[:] = delta2*lg.sigmoidGradient(z2[:,i:i+1])
      Theta1_grad = Theta1_grad + delta2.dot(X[i:i+1,:])
   J = J/(float(m))
   J = J + lamb*(np.sum(Theta1[:,1:]**2,axis=None, dtype=np.float64) + np.sum(Theta2[:,1:]**2,axis=None, dtype=np.float64))/(2*float(m))
   Theta1_grad[:,1:] = Theta1_grad[:,1:] + lamb*Theta1[:,1:]
   Theta2_grad[:,1:] = Theta2_grad[:,1:] + lamb*Theta2[:,1:]
   Theta1_grad = Theta1_grad/float(m)
   Theta2_grad = Theta2_grad/float(m)
   # unroll gradients
   grad = np.hstack((Theta1_grad.flatten(),Theta2_grad.flatten()))
   return (J, grad)

#-----------------BEGIN HEADERS-----------------
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
import csv
import scipy
np.set_printoptions(threshold=np.nan)
import contextlib
import pdb

@contextlib.contextmanager

def printoptions(*args, **kwargs):
  original = np.get_printoptions()
  np.set_printoptions(*args, **kwargs)
  yield 
  np.set_printoptions(**original)
#-----------------END HEADERS-----------------


#-----------------BEGIN FUNCTION 0-----------------
def readincsv(ratio_training_to_cv):

  data = []

  #Sort the data by num_labels, split into X and y
  print('Reading in data')
  with open('train.csv', 'rb') as csvfile:
    has_header = csv.Sniffer().has_header(csvfile.read(1024))
    csvfile.seek(0)  # rewind
    data_csv = csv.reader(csvfile, delimiter=',')
    if has_header:
      next(data_csv)
    for row in data_csv:
      data.append(row)
  data = np.array(data)

  #Set basic parameters
  x = data[:,1:]
  y = data[:,0]
  x = x.astype(float)
  y = y.astype(int)
  m, n = np.shape(x)
  
  #Set training and CV basic parameters
  m_train = int(m*ratio_training_to_cv)
  m_cv = m - m_train
  x_train = x[:m_train,:]
  x_cv = x[m_train:,:]
  y_train = y[:m_train]
  y_cv = y[m_train:]

  del(data)

  return (m, n, x, y, m_train, m_cv, x_train, x_cv, y_train, y_cv)
#-----------------END FUNCTION 0-----------------


#-----------------BEGIN FUNCTION 1-----------------
def randinitialize(L_in, L_out):
  w = np.zeros((L_out, 1 + L_in))
  epsilon_init = 0.12
  w = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

  return w
#-----------------END FUNCTION 1-----------------


#-----------------BEGIN FUNCTION 2-----------------
def sigmoid(lz):
  g = 1.0/(1.0+np.exp(-lz))

  return g
#-----------------END FUNCTION 2-----------------


#-----------------BEGIN FUNCTION 3-----------------
def sigmoidgradient(lz):
  g = np.multiply(sigmoid(lz),(1.0-sigmoid(lz)))

  return g
#-----------------END FUNCTION 3-----------------


#-----------------BEGIN FUNCTION 4-----------------
def nncostfunction3(ltheta_ravel, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg):
  ltheta1 = np.array(np.reshape(ltheta_ravel[:lhidden_layer_size * (linput_layer_size + 1)], (lhidden_layer_size, (linput_layer_size + 1))))
  ltheta2 = np.array(np.reshape(ltheta_ravel[lhidden_layer_size * (linput_layer_size + 1):], (lnum_labels, (lhidden_layer_size + 1))))
  ltheta1_grad = np.zeros((np.shape(ltheta1)))
  ltheta2_grad = np.zeros((np.shape(ltheta2)))
  y_matrix = []
  lm = np.shape(lx)[0]

  eye_matrix = np.eye(lnum_labels)
  for i in range(len(ly)):
    y_matrix.append(eye_matrix[int(ly[i]),:]) #The minus one as python is zero based
  y_matrix = np.array(y_matrix)

  a1 = np.hstack((np.ones((lm,1)), lx)).astype(float)
  z2 = sigmoid(ltheta1.dot(a1.T))
  a2 = (np.concatenate((np.ones((np.shape(z2)[1], 1)), z2.T), axis=1)).astype(float)
  a3 = sigmoid(ltheta2.dot(a2.T))
  h = a3
  J_unreg = 0
  J = 0
  if 0 in (h):
    #print("Yes 0")
    h[h==0] = 1e-6
  if 1 in (h):
    #print("Yes 1")
    h[h==1] = 1 - 1e-6
  J_unreg = (1.0/float(lm))*np.sum(\
  -np.multiply(y_matrix,np.log(h.T))\
  -np.multiply((1-y_matrix),np.log(1-h.T))\
  ,axis=None)
  J = J_unreg + (llambda_reg/(2.0*float(lm)))*\
  (np.sum(\
  np.multiply(ltheta1[:,1:],ltheta1[:,1:])\
  ,axis=None)+np.sum(\
  np.multiply(ltheta2[:,1:],ltheta2[:,1:])\
  ,axis=None))

  delta3 = a3.T - y_matrix
  delta2 = np.multiply((delta3.dot(ltheta2[:,1:])), (sigmoidgradient(ltheta1.dot(a1.T))).T)
  cdelta2 = ((a2.T).dot(delta3)).T
  cdelta1 = ((a1.T).dot(delta2)).T

  ltheta1_grad = (1.0/float(lm))*cdelta1
  ltheta2_grad = (1.0/float(lm))*cdelta2

  theta1_hold = ltheta1[:,:]
  theta2_hold = ltheta2[:,:]
  theta1_hold[:,0] = 0
  theta2_hold[:,0] = 0
  ltheta1_grad = ltheta1_grad + (llambda_reg/float(lm))*theta1_hold
  ltheta2_grad = ltheta2_grad + (llambda_reg/float(lm))*theta2_hold
  thetagrad_ravel = np.hstack((np.ravel(ltheta1_grad), np.ravel(ltheta2_grad)))

  return (J, thetagrad_ravel) 
#-----------------END FUNCTION 4-----------------


#-----------------BEGIN FUNCTION 4b-----------------
def nncostfunction4(ltheta_ravel, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lnum_labels, lx, ly, llambda_reg):
  ltheta1 = np.array(np.reshape(ltheta_ravel[0:lhidden_layer_size1*(linput_layer_size+1)], ((lhidden_layer_size1, linput_layer_size + 1))))
  ltheta2 = np.array(np.reshape(ltheta_ravel[lhidden_layer_size1*(linput_layer_size+1):(lhidden_layer_size1*(linput_layer_size+1)+lhidden_layer_size2*(lhidden_layer_size1+1))], ((lhidden_layer_size2, lhidden_layer_size1 + 1))))
  ltheta3 = np.array(np.reshape(ltheta_ravel[(lhidden_layer_size1*(linput_layer_size+1)+lhidden_layer_size2*(lhidden_layer_size1+1)):], ((lnum_labels, lhidden_layer_size2 + 1))))
  ltheta1_grad = np.zeros((np.shape(ltheta1)))
  ltheta2_grad = np.zeros((np.shape(ltheta2)))
  ltheta3_grad = np.zeros((np.shape(ltheta3)))
  y_matrix = []
  lm = np.shape(lx)[0]

  eye_matrix = np.eye(lnum_labels)
  for i in range(len(ly)):
    y_matrix.append(eye_matrix[int(ly[i]),:]) #The minus one as python is zero based
  y_matrix = np.array(y_matrix)

  a1 = np.hstack((np.ones((lm,1)), lx)).astype(float)
  z2 = sigmoid(ltheta1.dot(a1.T))
  a2 = (np.concatenate((np.ones((np.shape(z2)[1], 1)), z2.T), axis=1)).astype(float)
  z3 = sigmoid(ltheta2.dot(a2.T))
  a3 = (np.concatenate((np.ones((np.shape(z3)[1], 1)), z3.T), axis=1)).astype(float)
  a4 = sigmoid(ltheta3.dot(a3.T))

  h = a4
  J_unreg = 0
  J = 0
  if 0 in (h):
    #print("Yes 0")
    h[h==0] = 1e-6
  if 1 in (h):
    #print("Yes 1")
    h[h==1] = 1 - 1e-6
  J_unreg = (1.0/float(lm))*np.sum(\
  -np.multiply(y_matrix,np.log(h.T))\
  -np.multiply((1-y_matrix),np.log(1-h.T))\
  ,axis=None)
  J = J_unreg + (llambda_reg/(2.0*float(lm)))*\
  (np.sum(\
  np.multiply(ltheta1[:,1:],ltheta1[:,1:])\
  ,axis=None)+np.sum(\
  np.multiply(ltheta2[:,1:],ltheta2[:,1:])\
  ,axis=None)+np.sum(\
  np.multiply(ltheta3[:,1:],ltheta3[:,1:])\
  ,axis=None))

  delta4 = a4.T - y_matrix
  delta3 = np.multiply((delta4.dot(ltheta3[:,1:])), (sigmoidgradient(ltheta2.dot(a2.T))).T)
  delta2 = np.multiply((delta3.dot(ltheta2[:,1:])), (sigmoidgradient(ltheta1.dot(a1.T))).T)
  cdelta3 = ((a3.T).dot(delta4)).T
  cdelta2 = ((a2.T).dot(delta3)).T
  cdelta1 = ((a1.T).dot(delta2)).T

  ltheta1_grad = (1.0/float(lm))*cdelta1
  ltheta2_grad = (1.0/float(lm))*cdelta2
  ltheta3_grad = (1.0/float(lm))*cdelta3

  theta1_hold = ltheta1[:,:]
  theta2_hold = ltheta2[:,:]
  theta3_hold = ltheta3[:,:]
  theta1_hold[:,0] = 0
  theta2_hold[:,0] = 0
  theta3_hold[:,0] = 0
  ltheta1_grad = ltheta1_grad + (llambda_reg/float(lm))*theta1_hold
  ltheta2_grad = ltheta2_grad + (llambda_reg/float(lm))*theta2_hold
  ltheta3_grad = ltheta3_grad + (llambda_reg/float(lm))*theta3_hold

  thetagrad_ravel = np.hstack((np.ravel(ltheta1_grad), np.ravel(ltheta2_grad), np.ravel(ltheta3_grad)))

  return (J, thetagrad_ravel)
#-----------------END FUNCTION 4b-----------------


#-----------------BEGIN FUNCTION 4c-----------------
def nncostfunction5(ltheta_ravel, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lhidden_layer_size3, lnum_labels, lx, ly, llambda_reg):
  ltheta1 = np.array(np.reshape(ltheta_ravel[0:lhidden_layer_size1*(linput_layer_size+1)], ((lhidden_layer_size1, linput_layer_size + 1))))
  ltheta2 = np.array(np.reshape(ltheta_ravel[lhidden_layer_size1*(linput_layer_size+1):lhidden_layer_size1*(linput_layer_size+1)+lhidden_layer_size2*(lhidden_layer_size1+1)], ((lhidden_layer_size2, lhidden_layer_size1 + 1))))
  ltheta3 = np.array(np.reshape(ltheta_ravel[lhidden_layer_size1*(linput_layer_size+1)+lhidden_layer_size2*(lhidden_layer_size1+1):lhidden_layer_size1*(linput_layer_size+1)+lhidden_layer_size2*(lhidden_layer_size1+1)+lhidden_layer_size3*(lhidden_layer_size2+1)], ((lhidden_layer_size3, lhidden_layer_size2 + 1))))
  ltheta4 = np.array(np.reshape(ltheta_ravel[lhidden_layer_size1*(linput_layer_size+1)+lhidden_layer_size2*(lhidden_layer_size1+1)+lhidden_layer_size3*(lhidden_layer_size2+1):], ((lnum_labels, lhidden_layer_size3 + 1))))
  ltheta1_grad = np.zeros((np.shape(ltheta1)))
  ltheta2_grad = np.zeros((np.shape(ltheta2)))
  ltheta3_grad = np.zeros((np.shape(ltheta3)))
  ltheta4_grad = np.zeros((np.shape(ltheta4)))
  y_matrix = []
  lm = np.shape(lx)[0]

  eye_matrix = np.eye(lnum_labels)
  for i in range(len(ly)):
    y_matrix.append(eye_matrix[int(ly[i]),:]) #The minus one as python is zero based
  y_matrix = np.array(y_matrix)

  a1 = np.hstack((np.ones((lm,1)), lx)).astype(float)
  z2 = sigmoid(ltheta1.dot(a1.T))
  a2 = (np.concatenate((np.ones((np.shape(z2)[1], 1)), z2.T), axis=1)).astype(float)
  z3 = sigmoid(ltheta2.dot(a2.T))
  a3 = (np.concatenate((np.ones((np.shape(z3)[1], 1)), z3.T), axis=1)).astype(float)
  z4 = sigmoid(ltheta3.dot(a3.T))
  a4 = (np.concatenate((np.ones((np.shape(z4)[1], 1)), z4.T), axis=1)).astype(float)
  a5 = sigmoid(ltheta4.dot(a4.T))

  h = a5
  J_unreg = 0
  J = 0
  if 0 in (h):
    #print("Yes 0")
    h[h==0] = 1e-6
  if 1 in (h):
    #print("Yes 1")
    h[h==1] = 1 - 1e-6
  J_unreg = (1.0/float(lm))*np.sum(\
  -np.multiply(y_matrix,np.log(h.T))\
  -np.multiply((1-y_matrix),np.log(1-h.T))\
  ,axis=None)
  J = J_unreg + (llambda_reg/(2.0*float(lm)))*\
  (np.sum(\
  np.multiply(ltheta1[:,1:],ltheta1[:,1:])\
  ,axis=None)+np.sum(\
  np.multiply(ltheta2[:,1:],ltheta2[:,1:])\
  ,axis=None)+np.sum(\
  np.multiply(ltheta3[:,1:],ltheta3[:,1:])\
  ,axis=None)+np.sum(\
  np.multiply(ltheta4[:,1:],ltheta4[:,1:])\
  ,axis=None))

  delta5 = a5.T - y_matrix
  delta4 = np.multiply((delta5.dot(ltheta4[:,1:])), (sigmoidgradient(ltheta3.dot(a3.T))).T)
  delta3 = np.multiply((delta4.dot(ltheta3[:,1:])), (sigmoidgradient(ltheta2.dot(a2.T))).T)
  delta2 = np.multiply((delta3.dot(ltheta2[:,1:])), (sigmoidgradient(ltheta1.dot(a1.T))).T)
  cdelta4 = ((a4.T).dot(delta5)).T
  cdelta3 = ((a3.T).dot(delta4)).T
  cdelta2 = ((a2.T).dot(delta3)).T
  cdelta1 = ((a1.T).dot(delta2)).T

  ltheta1_grad = (1.0/float(lm))*cdelta1
  ltheta2_grad = (1.0/float(lm))*cdelta2
  ltheta3_grad = (1.0/float(lm))*cdelta3
  ltheta4_grad = (1.0/float(lm))*cdelta4

  theta1_hold = ltheta1[:,:]
  theta2_hold = ltheta2[:,:]
  theta3_hold = ltheta3[:,:]
  theta4_hold = ltheta4[:,:]
  theta1_hold[:,0] = 0
  theta2_hold[:,0] = 0
  theta3_hold[:,0] = 0
  theta4_hold[:,0] = 0
  ltheta1_grad = ltheta1_grad + (llambda_reg/float(lm))*theta1_hold
  ltheta2_grad = ltheta2_grad + (llambda_reg/float(lm))*theta2_hold
  ltheta3_grad = ltheta3_grad + (llambda_reg/float(lm))*theta3_hold
  ltheta4_grad = ltheta4_grad + (llambda_reg/float(lm))*theta4_hold

  thetagrad_ravel = np.hstack((np.ravel(ltheta1_grad), np.ravel(ltheta2_grad), np.ravel(ltheta3_grad), np.ravel(ltheta4_grad)))

  return (J, thetagrad_ravel)
#-----------------END FUNCTION 4c-----------------


#-----------------BEGIN FUNCTION 5-----------------
def predict3(ltheta1, ltheta2, x):
  m, n = np.shape(x)
  p = np.zeros(m)

  h1 = sigmoid((np.hstack((np.ones((m,1)),x.astype(float)))).dot(ltheta1.T))
  h2 = sigmoid((np.hstack((np.ones((m,1)),h1))).dot(ltheta2.T))

  for i in range(0,np.shape(h2)[0]):
    p[i] = np.argmax(h2[i,:])
  
  return p
#-----------------END FUNCTION 5-----------------


#-----------------BEGIN FUNCTION 5b-----------------
def predict4(ltheta1, ltheta2, ltheta3, x):
  m, n = np.shape(x)
  p = np.zeros(m)

  h1 = sigmoid((np.hstack((np.ones((m,1)),x.astype(float)))).dot(ltheta1.T))
  h2 = sigmoid((np.hstack((np.ones((m,1)),h1))).dot(ltheta2.T))
  h3 = sigmoid((np.hstack((np.ones((m,1)),h2))).dot(ltheta3.T))

  for i in range(0,np.shape(h3)[0]):
    p[i] = np.argmax(h3[i,:])

  return p
#-----------------END FUNCTION 5b-----------------


#-----------------BEGIN FUNCTION 5c-----------------
def predict5(ltheta1, ltheta2, ltheta3, ltheta4, x):
  m, n = np.shape(x)
  p = np.zeros(m)

  h1 = sigmoid((np.hstack((np.ones((m,1)),x.astype(float)))).dot(ltheta1.T))
  h2 = sigmoid((np.hstack((np.ones((m,1)),h1))).dot(ltheta2.T))
  h3 = sigmoid((np.hstack((np.ones((m,1)),h2))).dot(ltheta3.T))
  h4 = sigmoid((np.hstack((np.ones((m,1)),h3))).dot(ltheta4.T))

  for i in range(0,np.shape(h4)[0]):
    p[i] = np.argmax(h4[i,:])

  return p
#-----------------END FUNCTION 5c-----------------


#-----------------BEGIN FUNCTION 7-----------------
def debuginitialweights(layerasize, layerbsize):
  np.random.seed(seed=2)
  w = np.random.rand(layerasize,1+layerbsize)/10.0
  np.random.seed(seed=None)
  return w
#-----------------END FUNCTION 7-----------------


#-----------------BEGIN FUNCTION 8-----------------
def smallNN3(llambda_reg):
  #Some initial variables
  llambda_reg = 0
  linput_layer_size = 3
  lhidden_layer_size1 = 5
  lnum_labels = 3
  lm = 5
  ly = np.zeros(lm)

  # We generate some 'random' test data
  ltheta1 = debuginitialweights(lhidden_layer_size1, linput_layer_size);
  ltheta2 = debuginitialweights(lnum_labels, lhidden_layer_size1);
  #Reusing debugInitializeWeights to generate X
  lx = debuginitialweights(lm, linput_layer_size-1);
  np.random.seed(seed=1)
  for i in range(0,lm):
    ly[i] = np.random.randint(1,lnum_labels) #1 + mod(1:m, lnum_labels)';
  np.random.seed(seed=None)

  ltheta_ravel = np.hstack((np.ravel(ltheta1), np.ravel(ltheta2)))
  
  cost, grad = nncostfunction3(ltheta_ravel, linput_layer_size, lhidden_layer_size1, lnum_labels, lx, ly, llambda_reg);

  numgrad = np.zeros(len(ltheta_ravel))
  perturb = np.zeros(len(ltheta_ravel))
  gradstep = 1e-4;
  for p in range(0,len(ltheta_ravel)):
    perturb[p] = gradstep
    loss1, lk1 = nncostfunction3(ltheta_ravel - perturb, linput_layer_size, lhidden_layer_size1, lnum_labels, lx, ly, llambda_reg);
    loss2, lk2 = nncostfunction3(ltheta_ravel + perturb, linput_layer_size, lhidden_layer_size1, lnum_labels, lx, ly, llambda_reg);
    numgrad[p] = (loss2 - loss1)/(2.0*gradstep)
    perturb[p] = 0

  showdiff = np.hstack((numgrad, grad))
  print(numgrad)
  print(grad)
  print('The above two columns you get should be very similar. (Right-Your Numerical Gradient, Bottom-Analytical Gradient)\n\n')
  ldiff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
  print('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).\n' + 'Relative Difference: ' + str(ldiff))
#-----------------END FUNCTION 8-----------------


#-----------------BEGIN FUNCTION 8b-----------------
def smallNN4(llambda_reg):
  #Some initial variables
  llambda_reg = 0
  linput_layer_size = 3
  lhidden_layer_size1 = 5
  lhidden_layer_size2 = 5
  lnum_labels = 3
  lm = 5
  ly = np.zeros(lm)

  # We generate some 'random' test data
  ltheta1 = debuginitialweights(lhidden_layer_size1, linput_layer_size);
  ltheta2 = debuginitialweights(lhidden_layer_size2, lhidden_layer_size1);
  ltheta3 = debuginitialweights(lnum_labels, lhidden_layer_size2);
  #Reusing debugInitializeWeights to generate X
  lx = debuginitialweights(lm, linput_layer_size-1);
  np.random.seed(seed=1)
  for i in range(0,lm):
    ly[i] = np.random.randint(1,lnum_labels) #1 + mod(1:m, lnum_labels)';
  np.random.seed(seed=None)

  ltheta_ravel = np.hstack((np.ravel(ltheta1), np.ravel(ltheta2), np.ravel(ltheta3)))

  cost, grad = nncostfunction4(ltheta_ravel, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lnum_labels, lx, ly, llambda_reg);

  numgrad = np.zeros(len(ltheta_ravel))
  perturb = np.zeros(len(ltheta_ravel))
  gradstep = 1e-4;
  for p in range(0,len(ltheta_ravel)):
    perturb[p] = gradstep
    loss1, lk1 = nncostfunction4(ltheta_ravel - perturb, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lnum_labels, lx, ly, llambda_reg);
    loss2, lk2 = nncostfunction4(ltheta_ravel + perturb, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lnum_labels, lx, ly, llambda_reg);
    numgrad[p] = (loss2 - loss1)/(2.0*gradstep)
    perturb[p] = 0

  showdiff = np.hstack((numgrad, grad))
  print(numgrad)
  print(grad)
  print('The above two columns you get should be very similar. (Right-Your Numerical Gradient, Bottom-Analytical Gradient)\n\n')
  ldiff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
  print('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).\n' + 'Relative Difference: ' + str(ldiff))
#-----------------END FUNCTION 8b-----------------


#-----------------BEGIN FUNCTION 8c-----------------
def smallNN5(llambda_reg):
  #Some initial variables
  llambda_reg = 0
  linput_layer_size = 3
  lhidden_layer_size1 = 5
  lhidden_layer_size2 = 5
  lhidden_layer_size3 = 5
  lnum_labels = 3
  lm = 5
  ly = np.zeros(lm)

  # We generate some 'random' test data
  ltheta1 = debuginitialweights(lhidden_layer_size1, linput_layer_size);
  ltheta2 = debuginitialweights(lhidden_layer_size2, lhidden_layer_size1);
  ltheta3 = debuginitialweights(lhidden_layer_size3, lhidden_layer_size2);
  ltheta4 = debuginitialweights(lnum_labels, lhidden_layer_size3);
  #Reusing debugInitializeWeights to generate X
  lx = debuginitialweights(lm, linput_layer_size-1);
  np.random.seed(seed=1)
  for i in range(0,lm):
    ly[i] = np.random.randint(1,lnum_labels) #1 + mod(1:m, lnum_labels)';
  np.random.seed(seed=None)

  ltheta_ravel = np.hstack((np.ravel(ltheta1), np.ravel(ltheta2), np.ravel(ltheta3), np.ravel(ltheta4)))

  cost, grad = nncostfunction5(ltheta_ravel, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lhidden_layer_size3, lnum_labels, lx, ly, llambda_reg);

  numgrad = np.zeros(len(ltheta_ravel))
  perturb = np.zeros(len(ltheta_ravel))
  gradstep = 1e-4;
  for p in range(0,len(ltheta_ravel)):
    perturb[p] = gradstep
    loss1, lk1 = nncostfunction5(ltheta_ravel - perturb, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lhidden_layer_size3, lnum_labels, lx, ly, llambda_reg);
    loss2, lk2 = nncostfunction5(ltheta_ravel + perturb, linput_layer_size, lhidden_layer_size1, lhidden_layer_size2, lhidden_layer_size3, lnum_labels, lx, ly, llambda_reg);
    numgrad[p] = (loss2 - loss1)/(2.0*gradstep)
    perturb[p] = 0

  showdiff = np.hstack((numgrad, grad))
  print(numgrad)
  print(grad)
  print('The above two columns you get should be very similar. (Right-Your Numerical Gradient, Bottom-Analytical Gradient)\n\n')
  ldiff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
  print('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).\n' + 'Relative Difference: ' + str(ldiff))
#-----------------END FUNCTION 8c-----------------


#-----------------BEGIN FUNCTION 9-----------------
def readintestcsv():

  x_test = []
  with open('test.csv', 'rb') as csvfile2:
    has_header2 = csv.Sniffer().has_header(csvfile2.read(1024))
    csvfile2.seek(0)  # rewind
    data_csv2 = csv.reader(csvfile2, delimiter=',')
    if has_header2:
      next(data_csv2)
    for row in data_csv2:
      x_test.append(row)
  x_test = np.array(x_test)

  x_test = x_test.astype(float)
  m_test, n_test = np.shape(x_test)

  return(x_test, m_test, n_test)
#-----------------END FUNCTION 9-----------------


#-----------------BEGIN FUNCTION 10-----------------
def myoptimiser3(optimisation_jump, optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold):

  #Setting initial to lower thresholds
  hidden_layer_size1 = d1_reg_lower_threshold
  lambda_reg = lambda_reg_lower_threshold
  list_of_hidden_layer_size1 = []
  list_of_lambda_reg = []
  list_of_predicted = []

  while hidden_layer_size1 < d1_reg_upper_threshold:
    while lambda_reg < lambda_reg_upper_threshold:
      print("Now running: hidden_layer_size:" + str(hidden_layer_size1) + ", lambda_reg:" + str(lambda_reg))
      theta1_initial = randinitialize(input_layer_size, hidden_layer_size1);
      theta2_initial = randinitialize(hidden_layer_size1, num_labels);
      theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))
      fmin = scipy.optimize.minimize(fun=nncostfunction3, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': optimisation_iteration, 'disp': False})
      answer = fmin.x
      theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
      theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):], ((num_labels, hidden_layer_size1 + 1))))

      p_op = predict3(theta1, theta2, x_cv);
      correct_cv = [1 if a == b else 0 for (a, b) in zip(p_op,y_cv)]
      accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))

      list_of_hidden_layer_size1.append(hidden_layer_size1)
      list_of_lambda_reg.append(lambda_reg)
      list_of_predicted.append(accuracy_cv)
      print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)

      lambda_reg = lambda_reg*optimisation_jump
      del(p_op)
      del(correct_cv)
      del(accuracy_cv)
      del(fmin)
      del(answer)
      del(theta1)
      del(theta2)
      del(theta1_initial)
      del(theta2_initial)
      del(theta_initial_ravel)
    hidden_layer_size1 = int(hidden_layer_size1*optimisation_jump)
    lambda_reg = lambda_reg_lower_threshold

  a = np.argmax(list_of_predicted)

  return(list_of_hidden_layer_size1[a], list_of_lambda_reg[a]) 
#-----------------END FUNCTION 10-----------------


#-----------------BEGIN FUNCTION 10b-----------------
def myoptimiser4(optimisation_jump, optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold, d2_reg_lower_threshold, d2_reg_upper_threshold):

  #Setting initial to lower thresholds
  hidden_layer_size1 = d1_reg_lower_threshold
  hidden_layer_size2 = d2_reg_lower_threshold
  lambda_reg = lambda_reg_lower_threshold
  list_of_hidden_layer_size1 = []
  list_of_hidden_layer_size2 = []
  list_of_lambda_reg = []
  list_of_predicted = []

  while hidden_layer_size1 < d1_reg_upper_threshold:
    while hidden_layer_size2 < d2_reg_upper_threshold:
      while lambda_reg < lambda_reg_upper_threshold:
        print("Now running: hidden_layer_size:" + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", lambda_reg:" + str(lambda_reg))
        theta1_initial = randinitialize(input_layer_size, hidden_layer_size1);
        theta2_initial = randinitialize(hidden_layer_size1, hidden_layer_size2);
        theta3_initial = randinitialize(hidden_layer_size2, num_labels);
        theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial), np.ravel(theta3_initial)))
        fmin = scipy.optimize.minimize(fun=nncostfunction4, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': optimisation_iteration, 'disp': False})
        answer = fmin.x
        theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
        theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):(hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1))], ((hidden_layer_size2, hidden_layer_size1 + 1))))
        theta3 = np.array(np.reshape(answer[(hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)):], ((num_labels, hidden_layer_size2 + 1)))) 

        p_op = predict4(theta1, theta2, theta3, x_cv);
        correct_cv = [1 if a == b else 0 for (a, b) in zip(p_op,y_cv)]
        accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))

        list_of_hidden_layer_size1.append(hidden_layer_size1)
        list_of_hidden_layer_size2.append(hidden_layer_size2)
        list_of_lambda_reg.append(lambda_reg)
        list_of_predicted.append(accuracy_cv)
        print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)

        lambda_reg = lambda_reg*optimisation_jump
        del(p_op)
        del(correct_cv)
        del(accuracy_cv)
        del(fmin)
        del(answer)
        del(theta1)
        del(theta2)
        del(theta3)
        del(theta1_initial)
        del(theta2_initial)
        del(theta3_initial)
        del(theta_initial_ravel)
      hidden_layer_size2 = int(hidden_layer_size2*optimisation_jump)
      lambda_reg = lambda_reg_lower_threshold
    hidden_layer_size1 = int(hidden_layer_size1*optimisation_jump)
    hidden_layer_size2 = d2_reg_lower_threshold
    lambda_reg = lambda_reg_lower_threshold

  a = np.argmax(list_of_predicted)

  return(list_of_hidden_layer_size1[a], list_of_hidden_layer_size2[a], list_of_lambda_reg[a])
#-----------------END FUNCTION 10b-----------------


#-----------------BEGIN FUNCTION 10c-----------------
def myoptimiser5(optimisation_jump, optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold, d2_reg_lower_threshold, d2_reg_upper_threshold, d3_reg_lower_threshold, d3_reg_upper_threshold):

  #Setting initial to lower thresholds
  hidden_layer_size1 = d1_reg_lower_threshold
  hidden_layer_size2 = d2_reg_lower_threshold
  hidden_layer_size3 = d3_reg_lower_threshold
  lambda_reg = lambda_reg_lower_threshold
  list_of_hidden_layer_size1 = []
  list_of_hidden_layer_size2 = []
  list_of_hidden_layer_size3 = []
  list_of_lambda_reg = []
  list_of_predicted = []

  while hidden_layer_size1 < d1_reg_upper_threshold:
    while hidden_layer_size2 < d2_reg_upper_threshold:
      while hidden_layer_size3 < d3_reg_upper_threshold:
        while lambda_reg < lambda_reg_upper_threshold:
          print("Now running: hidden_layer_size:" + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", " + str(hidden_layer_size3) + ", lambda_reg:" + str(lambda_reg))
          theta1_initial = randinitialize(input_layer_size, hidden_layer_size1);
          theta2_initial = randinitialize(hidden_layer_size1, hidden_layer_size2);
          theta3_initial = randinitialize(hidden_layer_size2, hidden_layer_size3);
          theta4_initial = randinitialize(hidden_layer_size3, num_labels);
          theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial), np.ravel(theta3_initial), np.ravel(theta4_initial)))
          fmin = scipy.optimize.minimize(fun=nncostfunction5, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, hidden_layer_size2, hidden_layer_size3, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': optimisation_iteration, 'disp': False})
          answer = fmin.x
          theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
          theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):(hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1))], ((hidden_layer_size2, hidden_layer_size1 + 1))))
          theta3 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1):hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)+hidden_layer_size3*(hidden_layer_size2+1)], ((hidden_layer_size3, hidden_layer_size2 + 1))))
          theta4 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)+hidden_layer_size3*(hidden_layer_size2+1):], ((num_labels, hidden_layer_size3 + 1))))

          p_op = predict5(theta1, theta2, theta3, theta4, x_cv);
          correct_cv = [1 if a == b else 0 for (a, b) in zip(p_op,y_cv)]
          accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))

          list_of_hidden_layer_size1.append(hidden_layer_size1)
          list_of_hidden_layer_size2.append(hidden_layer_size2)
          list_of_hidden_layer_size3.append(hidden_layer_size3)
          list_of_lambda_reg.append(lambda_reg)
          list_of_predicted.append(accuracy_cv)
          print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)

          lambda_reg = lambda_reg*optimisation_jump
          del(p_op)
          del(correct_cv)
          del(accuracy_cv)
          del(fmin)
          del(answer)
          del(theta1)
          del(theta2)
          del(theta3)
          del(theta4)
          del(theta1_initial)
          del(theta2_initial)
          del(theta3_initial)
          del(theta4_initial)
          del(theta_initial_ravel)
        hidden_layer_size3 = int(hidden_layer_size3*optimisation_jump)
        lambda_reg = lambda_reg_lower_threshold
      hidden_layer_size2 = int(hidden_layer_size2*optimisation_jump)
      hidden_layer_size3 = d3_reg_lower_threshold
      lambda_reg = lambda_reg_lower_threshold
    hidden_layer_size1 = int(hidden_layer_size1*optimisation_jump)
    hidden_layer_size2 = d2_reg_lower_threshold
    hidden_layer_size3 = d3_reg_lower_threshold
    lambda_reg = lambda_reg_lower_threshold

  a = np.argmax(list_of_predicted)

  return(list_of_hidden_layer_size1[a], list_of_hidden_layer_size2[a], list_of_hidden_layer_size3[a], list_of_lambda_reg[a])
#-----------------END FUNCTION 10c-----------------

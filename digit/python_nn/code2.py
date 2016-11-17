#-----------------BEGIN HEADERS-----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import sys
import csv
import scipy
np.set_printoptions(threshold=np.nan)

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
	original = np.get_printoptions()
	np.set_printoptions(*args, **kwargs)
	yield 
	np.set_printoptions(**original)
#-----------------END HEADERS-----------------


#-----------------BEGIN FUNCTION 1-----------------
def randinitialize(L_in, L_out):
	W = np.zeros((L_out, 1 + L_in))
	epsilon_init = 0.12
	W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

	return W
#-----------------END FUNCTION 1-----------------


#-----------------BEGIN FUNCTION 2-----------------
def sigmoid(lz):
	g = 1.0/(1.0 + np.exp(-lz))

	return g
#-----------------END FUNCTION 2-----------------


#-----------------BEGIN FUNCTION 3-----------------
def sigmoidgradient(lz):
	g = np.multiply(sigmoid(lz),(1-sigmoid(lz)))

	return g
#-----------------END FUNCTION 3-----------------


#-----------------BEGIN FUNCTION 4-----------------
def nncostfunction(ltheta_ravel, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg):
	ltheta1 = np.matrix(np.reshape(ltheta_ravel[:lhidden_layer_size * (linput_layer_size + 1)], (lhidden_layer_size, (linput_layer_size + 1))))
	ltheta2 = np.matrix(np.reshape(ltheta_ravel[lhidden_layer_size * (linput_layer_size + 1):], (num_labels, (lhidden_layer_size + 1))))
	theta1_grad = np.zeros((np.shape(ltheta1)))
	theta2_grad = np.zeros((np.shape(ltheta2)))
	y_matrix = []
	m = np.shape(lx)[0]

	#print(ltheta2[8,8])
	#exit()

	eye_matrix = np.eye(lnum_labels)
	for i in range(len(y)):
		y_matrix.append(eye_matrix[int(y[i])-1,:]) #The minus one as python is zero based
	y_matrix = np.array(y_matrix)

	#print(y[0:5])
	#print(y_matrix[0:5,:])
	#print(np.shape(x))
	#exit()
	a1 = np.hstack((np.ones((m,1)), x)).astype(float)
	#a1 = (np.concatenate((np.ones((m, 1)), x), axis=1)).astype(float)
	z2 = sigmoid(ltheta1.dot(a1.T))
	a2 = (np.concatenate((np.ones((np.shape(z2)[1], 1)), z2.T), axis=1)).astype(float)
	a3 = sigmoid(ltheta2.dot(a2.T))
	h = a3

	#print(x[0:30,99:120])
	#print(ltheta1[0:5,0:2])
	#print(a1[0:30,99:120])
	#print(np.std(ltheta1))
	#print(np.std(ltheta2))
	#print((ltheta1.dot(a1.T))[0:5,0:2])
	#print(z2[0:5,0:2])
	#print(np.std(z2))
	#exit()

	#print(h[:,0])
	#print(np.shape(h))
	#exit()

	J_unreg = 0
	J = 0
	J_unreg = (1/float(m))*np.sum(\
	-np.multiply(y_matrix,np.log(h.T))\
	-np.multiply((1-y_matrix),np.log(1-h.T))\
	,axis=None)

	#print(-np.multiply(y_matrix,np.log(h.T))-np.multiply((1-y_matrix),np.log(1-h.T)))
	#print(np.sum(\
  #-np.multiply(y_matrix,np.log(h.T))\
  #-np.multiply((1-y_matrix),np.log(1-h.T))\
  #,axis=None))
	#print(np.multiply((1-y_matrix),np.log(1-h.T)))
	#exit()
	#print(J)
	#print(np.shape(J_unreg))
	#exit()

	J = J_unreg + (llambda_reg/(2*float(m)))*\
	(np.sum(\
	np.multiply(ltheta1[:,1:],ltheta1[:,1:])\
	,axis=None)+np.sum(\
	np.multiply(ltheta2[:,1:],ltheta2[:,1:])\
	,axis=None))

	print(J)
	#print(np.shape(J))
	#exit()

	delta3 = a3.T - y_matrix
	delta2 = np.multiply((delta3.dot(ltheta2[:,1:])), (sigmoidgradient(ltheta1.dot(a1.T))).T)
	cdelta2 = ((a2.T).dot(delta3)).T
	cdelta1 = ((a1.T).dot(delta2)).T

	theta1_grad = (1/float(m))*cdelta1
	theta2_grad = (1/float(m))*cdelta2

	theta1_hold = ltheta1
	theta2_hold = ltheta2
	theta1_hold[:,0] = 0;
	theta2_hold[:,0] = 0;
	theta1_grad = theta1_grad + (llambda_reg/float(m))*theta1_hold;
	theta2_grad = theta2_grad + (llambda_reg/float(m))*theta2_hold;

	#print(theta1_grad[0:10,0:10])
	#exit()

	thetagrad_ravel = np.concatenate((np.ravel(theta1_grad), np.ravel(theta2_grad)))
	#ltheta1_grad = np.matrix(np.reshape(thetagrad_ravel[:lhidden_layer_size * (linput_layer_size + 1)], (lhidden_layer_size, (linput_layer_size + 1))))
	#ltheta2_grad = np.matrix(np.reshape(thetagrad_ravel[lhidden_layer_size * (linput_layer_size + 1):], (num_labels, (lhidden_layer_size + 1))))
	#print(ltheta1_grad[7,8])
	#print(ltheta2_grad[7,8])
	#print(theta1_grad[7,8])
	#print(theta2_grad[7,8])
	#exit()

	#np.savetxt("../p_t1g.csv", theta1_grad, delimiter=",", fmt='%.10f')
	#np.savetxt("../p_t2g.csv", theta2_grad, delimiter=",", fmt='%.10f')
	#exit()


	return (J, thetagrad_ravel)	
#-----------------END FUNCTION 4-----------------


#-----------------BEGIN FUNCTION 5-----------------
def predict(ltheta1, ltheta2, x, m, num_labels):
	p = np.zeros(np.shape(x)[0])
	#p_val = np.zeros(np.shape(x)[0])
	h1 = sigmoid(np.hstack((np.ones((m,1)),x.astype(float)))*(ltheta1.T))
	h2 = sigmoid(np.hstack((np.ones((m,1)),h1))*(ltheta2.T))
	#h1 = sigmoid(((np.concatenate((np.ones((m,1)),x),axis=1)).astype(float)).dot((ltheta1.T).astype(float)))
	#h2 = sigmoid(((np.concatenate((np.ones((m,1)),h1),axis=1)).astype(float)).dot((ltheta2.T).astype(float)))
	#print(h2)
	#print(np.shape(h2))
	#exit()
	#print(h2[10000,:], fmt='%.5f')
	#print(x[0:10,0:10])
	#np.savetxt("../p_x.csv", x.astype(float), delimiter=",", fmt='%.0f')
	#np.savetxt("../p_t1.csv", ltheta1, delimiter=",", fmt='%.6f')
	#np.savetxt("../p_t2.csv", ltheta2, delimiter=",", fmt='%.6f')
	'''
	exit()
	with printoptions(precision=5, suppress=True):
		print(h2[0,:])
	#print("%.2f" % h2[10000,:])
	#print(np.argmax(h2[10000,:]))
	exit()
	'''
	for i in range(0,np.shape(h2)[0]):
		#p[i], p_val[i] = max(enumerate(h2[i,:]), key=operator.itemgetter(1))
		p[i] = np.argmax(h2[i,:])

	#print(p)
	#exit()
	
	return p
#-----------------END FUNCTION 5-----------------
def riseoverrun(J, theta):
	numgrad = np.zeros(np.shape(theta));
	perturb = np.zeros(np.shape(theta));
	e = 1e-4;
	for p in range(1,theta.size)
		perturb(p) = e
		loss1 = J(theta - perturb)
		loss2 = J(theta + perturb)
		numgrad[p] = (loss2 - loss1) / (2*e)
		perturb[p] = 0

	return numgrad
#-----------------BEGIN FUNCTION 6-----------------

#-----------------END FUNCTION 6-----------------


#-----------------BEGIN BODY-----------------
print("Started running")

## Setup the parameters you will use for this exercise
input_layer_size  = 784;  # 28x28 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 0 to 9
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
		#print ', '.join(row)
		data.append(row)
#data_sort = np.sort(data,axis=0)
#del(data)
data = np.array(data)
x = data[:,1:]
#print(x[26,69])
#print(x[0:30,99:120])
y = data[:,0]
y = y.astype(int)
for i in range(len(y)):
	if y[i] == 0:
		y[i] = 10

#print(y[0:10])
#exit()
#Set basic parameters
m, n = np.shape(x)
lambda_reg = 1.0

#print(m)
#print(n)
#exit()

#Randomly initalize weights for Theta
#print('Initializing weights')
#theta1 = randinitialize(input_layer_size, hidden_layer_size);
#theta2 = randinitialize(hidden_layer_size, num_labels);
#theta_ravel = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
#print(theta1[12,20])
#print(theta2[8,20])
#ltheta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
#ltheta2 = np.matrix(np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
#print(ltheta1[12,20])
#print(ltheta2[8,20])
#exit()

#Randomly initalize weights for Theta_initial
theta1_initial = np.genfromtxt('tt1.csv', delimiter=',')
theta2_initial = np.genfromtxt('tt2.csv', delimiter=',')
#theta1_initial = randinitialize(input_layer_size, hidden_layer_size);
#theta2_initial = randinitialize(hidden_layer_size, num_labels);
#np.savetxt("tc1.csv", theta1_initial, delimiter=",", fmt='%.5f')
#np.savetxt("tc2.csv", theta2_initial, delimiter=",", fmt='%.5f')
#exit()
theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))

numgrad = riseoverrun(J, theta);
exit()

#print(theta2_initial[0:5,0:5])
#exit()

print('Doing fminunc')
#Doing fminunc (Training)
#nncostfunction(theta_initial_ravel, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg);
#exit()
#fmin = scipy.optimize.minimize(fun=nncostfunction, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg), method='TNC', jac=True, options={'maxiter': 100})
fmin = scipy.optimize.minimize(fun=nncostfunction, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg), method='L-BFGS-B', jac=True, options={'maxiter': 10})
fmin
theta1 = np.matrix(np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
#print(theta1[0:3,0:3])
#exit()
#theta1 = theta1_initial
#theta2 = theta2_initial

#print(theta2)
#print(np.shape(theta2))
#exit()

p = predict(theta1, theta2, x, m, num_labels);
for i in range(len(y)):
	if y[i] == 10:
		y[i] = 0

#print("next")
#print(p[0:5])
#print(y[0:5])
#exit()
correct = [1 if a == b else 0 for (a, b) in zip(p,y)]  
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print 'accuracy = {0}%'.format(accuracy * 100)
exit()

#Sort the data by num_labels, split into X and y
print("Reading in test data")
x_test = []
with open('test.csv', 'rb') as csvfile2:
  has_header2 = csv.Sniffer().has_header(csvfile2.read(1024))
  csvfile2.seek(0)  # rewind
  data_csv2 = csv.reader(csvfile2, delimiter=',')
  if has_header2:
    next(data_csv2)
  for row in data_csv2:
    #print ', '.join(row)
    x_test.append(row)

m_test, n_test = np.shape(x_test)

p_test = predict(theta1, theta2, x_test, m_test, num_labels);
for i in range(len(p_test)):
	if p_test[i] == 10:
		p_test[i] = 0

imageid = []
for i in range(len(p_test)):
	imageid.append(i+1)

mysubmission = np.vstack((imageid,p_test)) 
np.set_printoptions(suppress=True)
np.savetxt("mytest.csv", mysubmission.T, delimiter=",", fmt='%.0f')
#-----------------END BODY-----------------

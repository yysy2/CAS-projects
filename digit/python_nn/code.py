#-----------------BEGIN HEADERS-----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import sys
import csv
import scipy
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

	eye_matrix = np.eye(lnum_labels)
	for i in range(len(y)):
		y_matrix.append(eye_matrix[int(y[i]),:])
	y_matrix = np.array(y_matrix)

	a1 = (np.concatenate((np.ones((m, 1)), x), axis=1)).astype(float)
	z2 = sigmoid(ltheta1.dot(a1.T))
	a2 = (np.concatenate((np.ones((np.shape(z2)[1], 1)), z2.T), axis=1)).astype(float)
	a3 = sigmoid(ltheta2.dot(a2.T))
	h = a3

	J_unreg = 0
	J = 0
	J_unreg = (1/m)*np.sum(\
	-np.multiply(y_matrix,np.log(h.T))\
	-np.multiply((1-y_matrix),np.log(1-h.T))\
	,axis=None)
	J = J_unreg + (llambda_reg/(2*m))*\
	(np.sum(\
	np.multiply(ltheta1[:,1:],ltheta1[:,1:])\
	,axis=None)+np.sum(\
	np.multiply(ltheta2[:,1:],ltheta2[:,1:])\
	,axis=None))

	delta3 = a3.T - y_matrix
	delta2 = np.multiply((delta3.dot(ltheta2[:,1:])), (sigmoidgradient(ltheta1.dot(a1.T))).T)
	cdelta2 = ((a2.T).dot(delta3)).T
	cdelta1 = ((a1.T).dot(delta2)).T

	theta1_grad = (1/m)*cdelta1
	theta2_grad = (1/m)*cdelta2

	theta1_hold = ltheta1
	theta2_hold = ltheta2
	theta1_hold[:,0] = 0;
	theta2_hold[:,0] = 0;
	theta1_grad = theta1_grad + (llambda_reg/m)*theta1_hold;
	theta2_grad = theta2_grad + (llambda_reg/m)*theta2_hold;

	thetagrad_ravel = np.concatenate((np.ravel(theta1_grad), np.ravel(theta2_grad)))
	#ltheta1_grad = np.matrix(np.reshape(thetagrad_ravel[:lhidden_layer_size * (linput_layer_size + 1)], (lhidden_layer_size, (linput_layer_size + 1))))
	#ltheta2_grad = np.matrix(np.reshape(thetagrad_ravel[lhidden_layer_size * (linput_layer_size + 1):], (num_labels, (lhidden_layer_size + 1))))
	#print(ltheta1_grad[7,8])
	#print(ltheta2_grad[7,8])
	#print(theta1_grad[7,8])
	#print(theta2_grad[7,8])
	#exit()

	return (J, thetagrad_ravel)	
#-----------------END FUNCTION 4-----------------


#-----------------BEGIN FUNCTION 5-----------------
def predict(ltheta1, ltheta2, x, m, num_labels):
	p = np.zeros((np.shape(x)[1], 1))
	h1 = sigmoid((np.concatenate((np.ones((m,1)),x),axis=1)).astype(float).dot((ltheta1.T).astype(float)))
	h2 = sigmoid((np.concatenate((np.ones((m,1)),h1),axis=1)).astype(float).dot((ltheta2.T).astype(float)))
	print(np.shape(h2))
	exit()
	#h1 = sigmoid([ones(m, 1) X] * Theta1')
	#h2 = sigmoid([ones(m, 1) h1] * Theta2')
	#[dummy, p] = max(h2, [], 2);

#-----------------END FUNCTION 5-----------------


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
data_sort = np.sort(data,axis=0)
del(data)
x = data_sort[:,1:]
y = data_sort[:,1]

#Set basic parameters
m, n = np.shape(x)
lambda_reg = 1

#Randomly initalize weights for Theta
print('Initializing weights')
theta1 = randinitialize(input_layer_size, hidden_layer_size);
theta2 = randinitialize(hidden_layer_size, num_labels);
theta_ravel = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
#print(theta1[12,20])
#print(theta2[8,20])
#ltheta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
#ltheta2 = np.matrix(np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
#print(ltheta1[12,20])
#print(ltheta2[8,20])
#exit()

#Randomly initalize weights for Theta_initial
theta1_initial = randinitialize(input_layer_size, hidden_layer_size);
theta2_initial = randinitialize(hidden_layer_size, num_labels);
theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))

print('Doing fminunc')
#Doing fminunc (Training)
#nncostfunction(nn_params_initial, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg);
fmin = scipy.optimize.minimize(fun=nncostfunction, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg), method='TNC', jac=True, options={'maxiter': 10})
fmin
theta1 = np.matrix(np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))

predict(theta1, theta2, x, m, num_labels);
#-----------------END BODY-----------------

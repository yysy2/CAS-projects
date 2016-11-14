#-----------------BEGIN HEADERS-----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import sys
import csv
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
	g = sigmoid(lz)*(1-sigmoid(lz))

	return g
#-----------------END FUNCTION 4-----------------


#-----------------BEGIN FUNCTION 4-----------------
def nncostfunction(ltheta1, ltheta2, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg):
	theta1_grad = np.zeros((np.shape(ltheta1)))
	theta2_grad = np.zeros((np.shape(ltheta2)))
	y_matrix = []
	m = len(lx[:,1])

	eye_matrix = np.eye(lnum_labels)
	for i in range(len(y)):
		y_matrix.append(eye_matrix[int(y[i]),:])
	y_matrix = np.array(y_matrix)

	a1 = (np.concatenate((np.ones((m, 1)), x), axis=1)).astype(float)
	z2 = sigmoid(ltheta1.dot(a1.T))
	a2 = (np.concatenate((np.ones((len(z2[1,:]), 1)), z2.T), axis=1)).astype(float)
	a3 = sigmoid(ltheta2.dot(a2.T))
	h = a3

	J_unreg = 0
	J = 0
	J_unreg = (1/m)*sum(sum(-y_matrix*np.log(h.T)-(1-y_matrix)*np.log(1-h.T)))
	J = J_unreg + (llambda_reg/(2*m))*(sum(sum(ltheta1[:,2:]*ltheta1[:,2:]))+sum(sum(ltheta2[:,2:]*ltheta2[:,2:])))
	delta3 = a3.T - y_matrix
	delta2 = (delta3.dot(ltheta2[:,2:]))*(sigmoidgradient(ltheta1.dot(a1.T))).T
	print(delta2)
	print(np.shape(delta2))
	exit()
	
#-----------------END FUNCTION 4-----------------


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

#Randomly initalize weights for Theta_initial
theta1_initial = randinitialize(input_layer_size, hidden_layer_size);
theta2_initial = randinitialize(hidden_layer_size, num_labels);

print('Doing fminunc')
#Doing fminunc (Training)
nncostfunction(theta1, theta2, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg);

#-----------------END BODY-----------------

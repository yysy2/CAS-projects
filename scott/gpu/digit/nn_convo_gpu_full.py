#-----------------BEGIN HEADERS-----------------
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
import csv
import scipy
np.set_printoptions(threshold=np.nan)
import contextlib
import pdb
import myfunctions as mf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from urllib import urlretrieve
import cPickle as pickle
import os
import gzip

import numpy as np
import theano

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.grid_search import GridSearchCV

@contextlib.contextmanager

def printoptions(*args, **kwargs):
  original = np.get_printoptions()
  np.set_printoptions(*args, **kwargs)
  yield 
  np.set_printoptions(**original)
#-----------------END HEADERS-----------------


#-----------------BEGIN BODY-----------------
print("Started running")

## Setup the parameters you will use for this exercise -- YOU WILL NEED TO SET THESE FLAGS BEFORE STARTING!!!
#############################################################################################################
##Basic flags
input_layer_size  = 784             # 28x28 Input Images of Digits
hidden_layer_size1 = 25             # hidden units, unless allow_optimisation = True
num_labels = 10                     # 10 labels, from 0 to 9
lambda_reg = 1.0                    # Regularisation parameter, allow_optimisation = True
ratio_training_to_cv = 0.7          # Sets the ratio of training to cv data

##Minimiser options
epochs = 10
#conv1d1
conv1d1_num_filters = 10
conv1d1_filter_size = 25
#maxpool1
maxpool1_pool_size = 5
#conv1d2
conv1d2_num_filters = 10
conv1d2_filter_size = 25
#maxpool2
maxpool2_pool_size = 5
#dropout1
dropout1_p = 0.5
#hidden1
hidden_layer_size1 = 256
#dropout2
dropout2_p = 0.5

##Optimisation options
allow_optimisation = True          # If True, will try to find best hidden layers and lambda. It will ignore inputted numbers. Only work if use_all_training_data = False and use_random_initialisation = True
only_optimisation = True           # If True, will exit after optimisation, only works if allow_optimisation = True
optimisation_epoch = 2         # Sets how many iterations when doing optimisation
#conv1d1
conv1d1_num_filters_lower = 5
conv1d1_num_filters_upper = 15
conv1d1_num_filters_skip = 5
conv1d1_filter_size_lower = 15
conv1d1_filter_size_upper = 25
conv1d1_filter_size_skip = 5
#maxpool1
maxpool1_pool_size_lower = 5
maxpool1_pool_size_upper = 10
maxpool1_pool_size_skip = 5
#conv1d2
conv1d2_num_filters_lower = 5
conv1d2_num_filters_upper = 15
conv1d2_num_filters_skip = 5
conv1d2_filter_size_lower = 15
conv1d2_filter_size_upper = 25
conv1d2_filter_size_skip = 5
#maxpool2
maxpool2_pool_size_lower = 5
maxpool2_pool_size_upper = 10
maxpool2_pool_size_skip = 5
#dropout1
dropout1_p_lower = 0.5
dropout1_p_upper = 0.5
dropout1_p_skip = 0.1
#hidden1
hidden_layer_size1_lower = 256
hidden_layer_size1_upper = 256
hidden_layer_size1_skip = 1
#dropout2
dropout2_p_lower = 0.5
dropout2_p_upper = 0.5
dropout2_p_skip = 0.1

##Output CSV file options
output_test_submission = False      # If True, will print out test data for submission


##Reading in data
#############################################################################################################
m, n, x, y, m_train, m_cv, x_train, x_cv, y_train, y_cv, x_raw = mf.readincsv(ratio_training_to_cv);
x_train = np.expand_dims(x_train.astype(np.float32),axis=1)
y_train = y_train.astype(np.int32)
x_cv = np.expand_dims(x_cv.astype(np.float32),axis=1)
y_cv = y_cv.astype(np.int32)

##Optimising d
#############################################################################################################
#The iterator name goes (iter)i, (type)c,p,h,d, (number)1,2, (ending) nf=num_filter, fs=filter_size
#accuracy_cv = mf.newnn(input_layer_size, conv1d1_num_filters, conv1d1_filter_size, maxpool1_pool_size, conv1d2_num_filters, conv1d2_filter_size, maxpool2_pool_size, dropout1_p, hidden_layer_size1, dropout2_p, num_labels, epochs, x_train, y_train, x_cv, y_cv);

'''
param_dist = {linput_layer_size,
              lconv1d1_num_filters,
              lconv1d1_filter_size,
              lmaxpool1_pool_size,
              lconv1d2_num_filters,
              lconv1d2_filter_size,
              lmaxpool2_pool_size,
              ldropout1_p,
              lhidden_layer_size1,
              ldropout2_p,
              lnum_labels,
              epochs,
              x_train,
              y_train,
              x_cv,
              y_cv
'''

net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv1d1', layers.Conv1DLayer),
            ('maxpool1', layers.MaxPool1DLayer),
            #('conv1d2', layers.Conv1DLayer),
            #('maxpool2', layers.MaxPool1DLayer),
            #('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            #('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, input_layer_size),
    # layer conv2d1
    conv1d1_num_filters=conv1d1_num_filters,
    conv1d1_filter_size=conv1d1_filter_size,
    conv1d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv1d1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=maxpool1_pool_size,
    # layer conv2d2
    #conv1d2_num_filters=lconv1d2_num_filters,
    #conv1d2_filter_size=lconv1d2_filter_size,
    #conv1d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    #maxpool2_pool_size=lmaxpool2_pool_size,
    # dropout1
    #dropout1_p=ldropout1_p,
    # dense
    dense_num_units=hidden_layer_size1,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    # dropout2
    #dropout2_p=ldropout2_p,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=num_labels,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )


param_dist = {'dense_num_units': [200, 300],
              'conv1d1_num_filters': [5, 10, 15]}
grid_search = GridSearchCV(net1, param_grid=param_dist)
grid_search.fit(x_train, y_train)
print(grid_search.grid_scores_)


exit()

'''
param_dist = {"input_layer_size": [input_layer_size, input_layer_size]
              "conv1d1_num_filters": sp_randint(conv1d1_num_filters_lower, conv1d1_num_filters_upper),
              "conv1d1_filter_size": sp_randint(conv1d1_filter_size_lower, conv1d1_filter_size_upper),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
'''

if allow_optimisation == True:
  print("Running optimisation")
  opti_details = []
  accuracy_cv_top = 0
  ic1nf = conv1d1_num_filters_lower
  ic1fs = conv1d1_filter_size_lower
  ip1 = maxpool1_pool_size_lower
  ic2nf = conv1d2_num_filters_lower
  ic2fs = conv1d2_filter_size_lower
  ip2 = maxpool2_pool_size_lower
  id1 = dropout1_p_lower
  ih1 = hidden_layer_size1_lower
  id2 = dropout2_p_lower
  while ic1nf < conv1d1_num_filters_upper:
    while ic1fs < conv1d1_filter_size_upper:
      while ip1 < maxpool1_pool_size_upper:
        while ic2nf < conv1d2_num_filters_upper:
          while ic2fs < conv1d2_filter_size_upper:
            while ip2 < maxpool2_pool_size_upper:
              while id1 < dropout1_p_upper:
                for ih1 in range(hidden_layer_size1_lower, hidden_layer_size1_upper, hidden_layer_size1_skip):
                  for id2 in range(dropout2_p_lower, dropout2_p_upper, dropout2_p_skip):
                    accuracy_cv = mf.newnn(input_layer_size, ic1nf, ic1fs, ip1, ic2nf, ic2fs, ip2, id1, ih1, id2, num_labels, optimisation_epoch, x_train, y_train, x_cv, y_cv);
                    print(str(ic1nf) + ', ' + str(ic1fs) + ', ' + str(ip1)+ ', ' + str(ic2nf) + ', ' + str(ic2fs) + ', ' + str(ip2) + ', ' + str(id1) + ', ' + str(ih1) + ', ' + str(id2) + ', ' + str(accuracy_cv))
                    if accuracy_cv > accuracy_cv_top:
                      print("new high: " + str(accuracy_cv))
                      accuracy_cv_top = accuracy_cv
                      del(opti_details)
                      opti_details = []
                      opti_details.append(ic1nf)
                      opti_details.append(ic1fs)
                      opti_details.append(ip1)
                      opti_details.append(ic2nf)
                      opti_details.append(ic2fs)
                      opti_details.append(ip2)
                      opti_details.append(id1)
                      opti_details.append(ih1)
                      opti_details.append(id2)
                      opti_details.append(accuracy_cv)

if only_optimisation == True:
  exit()

exit()
accuracy_cv = mf.newnn(input_layer_size, conv1d1_num_filters, conv1d1_filter_size, maxpool1_pool_size, conv1d2_num_filters, conv1d2_filter_size, maxpool2_pool_size, dropout1_p, hidden_layer_size1, dropout2_p, num_labels, epochs, x_train, y_train, x_cv, y_cv);


'''
net1 = NeuralNet(
  layers=[('input', layers.InputLayer),
          ('conv1d1', layers.Conv1DLayer),
          ('maxpool1', layers.MaxPool1DLayer),
          ('conv1d2', layers.Conv1DLayer),
          ('maxpool2', layers.MaxPool1DLayer),
          ('dropout1', layers.DropoutLayer),
          ('dense', layers.DenseLayer),
          ('dropout2', layers.DropoutLayer),
          ('output', layers.DenseLayer),
          ],
  # input layer
  input_shape=(None, 1, input_layer_size),
  # layer conv2d1
  conv1d1_num_filters=10,
  conv1d1_filter_size=25,
  conv1d1_nonlinearity=lasagne.nonlinearities.rectify,
  conv1d1_W=lasagne.init.GlorotUniform(),
  # layer maxpool1
  maxpool1_pool_size=5,
  # layer conv2d2
  conv1d2_num_filters=10,
  conv1d2_filter_size=25,
  conv1d2_nonlinearity=lasagne.nonlinearities.rectify,
  # layer maxpool2
  maxpool2_pool_size=5,
  # dropout1
  dropout1_p=0.5,
  # dense
  dense_num_units=256,
  dense_nonlinearity=lasagne.nonlinearities.rectify,
  # dropout2
  dropout2_p=0.5,
  # output
  output_nonlinearity=lasagne.nonlinearities.softmax,
  output_num_units=10,
  # optimization method params
  update=nesterov_momentum,
  update_learning_rate=0.01,
  update_momentum=0.9,
  max_epochs=20,
  verbose=1,
  )

nn = net1.fit(x_train, y_train)
p_train = net1.predict(x_train)
correct = [1 if a == b else 0 for (a, b) in zip(p_train,y_train)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print 'training set accuracy = {0}%'.format(accuracy * 100)
p_cv = net1.predict(x_cv)
correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
print 'training set accuracy = {0}%'.format(accuracy_cv * 100)
'''
exit()

#Optimising d and lambda
#############################################################################################################
if allow_optimisation == True:
  if use_all_training_data == True:
    print("Must set use_all_training_data = False for this to work")
    exit()
  elif use_random_initialisation == False:
    print("Must set use_random_initialisation = True for this to work")
  else:
    print('Doing optimisation')
    if number_of_layers == 3:
      del(hidden_layer_size1)
      del(lambda_reg)
      hidden_layer_size1, lambda_reg = mf.myoptimiser3(optimisation_jump, optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold);
    elif number_of_layers == 4:
      del(hidden_layer_size1)
      del(hidden_layer_size2)
      del(lambda_reg)
      hidden_layer_size1, hidden_layer_size2, lambda_reg = mf.myoptimiser4(optimisation_jump, optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold, d2_reg_lower_threshold, d2_reg_upper_threshold);
    elif number_of_layers == 5:
      del(hidden_layer_size1)
      del(hidden_layer_size2)
      del(lambda_reg)
      hidden_layer_size1, hidden_layer_size2, hidden_layer_size2, lambda_reg = mf.myoptimiser5(optimisation_jump, optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold, d2_reg_lower_threshold, d2_reg_upper_threshold, d3_reg_lower_threshold, d3_reg_upper_threshold);
    else:
      print("Number of layers must be 3, 4 or 5!!! :(")

if number_of_layers == 3:
  print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", lambda: " + str(lambda_reg))
elif number_of_layers == 4:
  print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", lambda: " + str(lambda_reg))
elif number_of_layers == 5:
  print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", " + str(hidden_layer_size3) + ", lambda: " + str(lambda_reg))
else:
  print("Number of layers must be 3, 4 or 5")
  exit()


if only_optimisation == True:
  exit()

#Randomly initalize weights for Theta_initial
#############################################################################################################
print('Initialising weights')
if use_random_initialisation == True:
  if number_of_layers == 3:
    theta1_initial = mf.randinitialize(input_layer_size, hidden_layer_size1);
    theta2_initial = mf.randinitialize(hidden_layer_size1, num_labels);
    theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))
  elif number_of_layers == 4:
    theta1_initial = mf.randinitialize(input_layer_size, hidden_layer_size1);
    theta2_initial = mf.randinitialize(hidden_layer_size1, hidden_layer_size2);
    theta3_initial = mf.randinitialize(hidden_layer_size2, num_labels);
    theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial), np.ravel(theta3_initial)))
  elif number_of_layers == 5:
    theta1_initial = mf.randinitialize(input_layer_size, hidden_layer_size1);
    theta2_initial = mf.randinitialize(hidden_layer_size1, hidden_layer_size2);
    theta3_initial = mf.randinitialize(hidden_layer_size2, hidden_layer_size3);
    theta4_initial = mf.randinitialize(hidden_layer_size3, num_labels);
    theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial), np.ravel(theta3_initial), np.ravel(theta4_initial)))
  else:
    print("Number of layers must be 3, 4 or 5")
    exit()
else:
  theta1_initial = np.genfromtxt('tt1.csv', delimiter=',')
  theta2_initial = np.genfromtxt('tt2.csv', delimiter=',')
  theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))

#Minimize nncostfunction
#############################################################################################################
print('Doing minimisation')
if use_all_training_data == True and number_of_layers == 3:
  fmin = scipy.optimize.minimize(fun=mf.nncostfunction3, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, num_labels, x, y, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})
  answer = fmin.x
  theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
  theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):], ((num_labels, hidden_layer_size1 + 1))))
elif use_all_training_data == True and number_of_layers == 4:
  fmin = scipy.optimize.minimize(fun=mf.nncostfunction4, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, x, y, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})
  answer = fmin.x
  theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
  theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)], ((hidden_layer_size2, hidden_layer_size1 + 1))))
  theta3 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1):], ((num_labels, hidden_layer_size2 + 1))))
elif use_all_training_data == True and number_of_layers == 5:
  fmin = scipy.optimize.minimize(fun=mf.nncostfunction5, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, hidden_layer_size2, hidden_layer_size3, num_labels, x, y, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})
  answer = fmin.x
  theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
  theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)], ((hidden_layer_size2, hidden_layer_size1 + 1))))
  theta3 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1):hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)+hidden_layer_size3*(hidden_layer_size2+1)], ((hidden_layer_size3, hidden_layer_size2 + 1))))
  theta4 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)+hidden_layer_size3*(hidden_layer_size2+1):], ((num_labels, hidden_layer_size3 + 1))))
elif use_all_training_data == False and number_of_layers == 3:
  fmin = scipy.optimize.minimize(fun=mf.nncostfunction3, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})
  answer = fmin.x
  theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
  theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):], ((num_labels, hidden_layer_size1 + 1))))
elif use_all_training_data == False and number_of_layers == 4:
  fmin = scipy.optimize.minimize(fun=mf.nncostfunction4, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})
  answer = fmin.x
  theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
  theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)], ((hidden_layer_size2, hidden_layer_size1 + 1))))
  theta3 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1):], ((num_labels, hidden_layer_size2 + 1))))
elif use_all_training_data == False and number_of_layers == 5:
  fmin = scipy.optimize.minimize(fun=mf.nncostfunction5, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size1, hidden_layer_size2, hidden_layer_size3, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})
  answer = fmin.x
  theta1 = np.array(np.reshape(answer[0:hidden_layer_size1*(input_layer_size+1)], ((hidden_layer_size1, input_layer_size + 1))))
  theta2 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1):hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)], ((hidden_layer_size2, hidden_layer_size1 + 1))))
  theta3 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1):hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)+hidden_layer_size3*(hidden_layer_size2+1)], ((hidden_layer_size3, hidden_layer_size2 + 1))))
  theta4 = np.array(np.reshape(answer[hidden_layer_size1*(input_layer_size+1)+hidden_layer_size2*(hidden_layer_size1+1)+hidden_layer_size3*(hidden_layer_size2+1):], ((num_labels, hidden_layer_size3 + 1))))
else:
  print("Error")
  exit()

#Doing predictions
#############################################################################################################
print('Doing predictions')
if use_all_training_data == True:
  p = mf.predict(theta1, theta2, x);
  correct = [1 if a == b else 0 for (a, b) in zip(p,y)]  
  accuracy = (sum(map(int, correct)) / float(len(correct)))  
  print 'training set accuracy = {0}%'.format(accuracy * 100)
else:
  if number_of_layers == 3:
    p = mf.predict3(theta1, theta2, x_train);
    print(p[0:10])
    print(y_train[0:10])
    correct = [1 if a == b else 0 for (a, b) in zip(p,y_train)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'training set accuracy = {0}%'.format(accuracy * 100)

    p_cv = mf.predict3(theta1, theta2, x_cv);
    correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
    accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
    print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)
  elif number_of_layers == 4:
    p = mf.predict4(theta1, theta2, theta3, x_train);
    print(p[0:10])
    print(y_train[0:10])
    correct = [1 if a == b else 0 for (a, b) in zip(p,y_train)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'training set accuracy = {0}%'.format(accuracy * 100)

    p_cv = mf.predict4(theta1, theta2, theta3, x_cv);
    correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
    accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
    print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)
  elif number_of_layers == 5:
    p = mf.predict5(theta1, theta2, theta3, theta4, x_train);
    print(p[0:10])
    print(y_train[0:10])
    correct = [1 if a == b else 0 for (a, b) in zip(p,y_train)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'training set accuracy = {0}%'.format(accuracy * 100)

    p_cv = mf.predict5(theta1, theta2, theta3, theta4, x_cv);
    correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
    accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
    print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)

#Processing test data
#############################################################################################################
if output_test_submission == True:
  print("Processing test data")
  if number_of_layers == 3:
    x_test, m_test, n_test = mf.readintestcsv();
    p_test = mf.predict3(theta1, theta2, x_test);
    imageid = []
    for i in range(len(p_test)):
      imageid.append(i+1)
    mysubmission = np.vstack((imageid,p_test))
    np.set_printoptions(suppress=True)
    np.savetxt("mytest_3layers.csv", mysubmission.T, delimiter=",", fmt='%.0f')
    print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", lambda: " + str(lambda_reg))
  elif number_of_layers == 4:
    x_test, m_test, n_test = mf.readintestcsv();
    p_test = mf.predict4(theta1, theta2, theta3, x_test);
    imageid = []
    for i in range(len(p_test)):
      imageid.append(i+1)
    mysubmission = np.vstack((imageid,p_test))
    np.set_printoptions(suppress=True)
    np.savetxt("mytest_4layers.csv", mysubmission.T, delimiter=",", fmt='%.0f')
    print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", lambda: " + str(lambda_reg))
  elif number_of_layers == 5:
    x_test, m_test, n_test = mf.readintestcsv();
    p_test = mf.predict5(theta1, theta2, theta3, theta4, x_test);
    imageid = []
    for i in range(len(p_test)):
      imageid.append(i+1)
    mysubmission = np.vstack((imageid,p_test))
    np.set_printoptions(suppress=True)
    np.savetxt("mytest_5layers.csv", mysubmission.T, delimiter=",", fmt='%.0f')
    print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", " + str(hidden_layer_size3) + ", lambda: " + str(lambda_reg))
  else:
    print("Number of layers must be 3, 4 or 5")
#-----------------END BODY-----------------

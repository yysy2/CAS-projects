#-----------------BEGIN HEADERS-----------------
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
import csv
import scipy
np.set_printoptions(threshold=np.nan)
import contextlib
import pdb
from random import shuffle
import myfunctions as mf

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
#input_layer_size  = 784             # 28x28 Input Images of Digits
hidden_layer_size1 = 1600             # hidden units, unless allow_optimisation = True
hidden_layer_size2 = 1600            # hidden units, unless allow_optimisation = True, ignored if number_of_layers = 3
hidden_layer_size3 = 1600            # hidden units, unless allow_optimisation = True, ignored if number_of_layers = 3 or 4
num_labels = 8                     # 10 labels, from 0 to 9
number_of_layers = 3                # Gives the number of layers in nn. 3, 4, 5 are available.
lambda_reg = 1.0                    # Regularisation parameter, allow_optimisation = True
ratio_training_to_cv = 0.7          # Sets the ratio of training to cv data
use_all_training_data = False       # If True, will use all training data instead of spliting into train and CV
colour = True                      # Select if we use RGB or greyscale

##Initialisation
use_random_initialisation = True    # If true, it will use random initialisation, if false, will use preset random values (FOR DEBUGGING ONLY) -- ONLY WORKS IF ALLOW_OPTIMISER = FALSE AND HIDDEN LAYER = 25 AND LAYERS = 3

##Gradient checking
use_gradient_checking = False        # If true, will turn on gradient checking (FOR DEBUGGING/FIRST RUN ONLY)
only_gradient_checking = False      # If true, will exit after gradient checking

##Minimiser options
iteration_number = 300               # Number of iterations
minimisation_method = "L-BFGS-B"    # Sets minimiser method, recommended L-BFGS-B or TNC
use_minimisation_display = True     # Sets whether we display iterations

##Optimisation options
allow_optimisation = False          # If True, will try to find best hidden layers and lambda. It will ignore inputted numbers. Only work if use_all_training_data = False and use_random_initialisation = True
only_optimisation = False           # If True, will exit after optimisation, only works if allow_optimisation = True
optimisation_iteration = 100         # Sets how many iterations when doing optimisation
optimisation_jump = 4.0             # Sets how multiplier
lambda_reg_lower_threshold = 5.0    # Sets the min lambda threshold for optimisation
lambda_reg_upper_threshold = 350.0  # Sets the max lambda threshold for optimisation
d1_reg_lower_threshold = 100         # Sets the min d1 threshold for optimisation
d1_reg_upper_threshold = 2700       # Sets the max d1 threshold for optimisation
d2_reg_lower_threshold = 100         # Sets the min d1 threshold for optimisation
d2_reg_upper_threshold = 2700       # Sets the max d1 threshold for optimisation
d3_reg_lower_threshold = 100         # Sets the min d1 threshold for optimisation
d3_reg_upper_threshold = 2700       # Sets the max d1 threshold for optimisation

##Output CSV file options
output_test_submission = False      # If True, will print out test data for submission

##Reading in data
#############################################################################################################
m, n, x, y, m_train, m_cv, x_train, x_cv, y_train, y_cv = mf.readincsv(ratio_training_to_cv, colour);
input_layer_size = n

#Gradient checking
#############################################################################################################
if use_gradient_checking == True:
  print('Doing gradient checking')
  if number_of_layers == 3:
    mf.smallNN3(lambda_reg);
  elif number_of_layers == 4:
    mf.smallNN4(lambda_reg);
  elif number_of_layers == 5:
    mf.smallNN5(lambda_reg);

if only_gradient_checking == True:
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
  p, h = mf.predict(theta1, theta2, x);
  correct = [1 if a == b else 0 for (a, b) in zip(p,y)]  
  accuracy = (sum(map(int, correct)) / float(len(correct)))  
  print 'training set accuracy = {0}%'.format(accuracy * 100)
else:
  if number_of_layers == 3:
    p, h = mf.predict3(theta1, theta2, x_train);
    print(p[0:10])
    print(y_train[0:10])
    correct = [1 if a == b else 0 for (a, b) in zip(p,y_train)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'training set accuracy = {0}%'.format(accuracy * 100)

    p_cv, h_cv = mf.predict3(theta1, theta2, x_cv);
    correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
    accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
    print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)
  elif number_of_layers == 4:
    p, h = mf.predict4(theta1, theta2, theta3, x_train);
    print(p[0:10])
    print(y_train[0:10])
    correct = [1 if a == b else 0 for (a, b) in zip(p,y_train)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'training set accuracy = {0}%'.format(accuracy * 100)

    p_cv, h_cv = mf.predict4(theta1, theta2, theta3, x_cv);
    correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
    accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
    print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)
  elif number_of_layers == 5:
    p, h = mf.predict5(theta1, theta2, theta3, theta4, x_train);
    print(p[0:10])
    print(y_train[0:10])
    correct = [1 if a == b else 0 for (a, b) in zip(p,y_train)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'training set accuracy = {0}%'.format(accuracy * 100)

    p_cv, h_cv = mf.predict5(theta1, theta2, theta3, theta4, x_cv);
    correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
    accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
    print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)

#Processing test data
#############################################################################################################
if output_test_submission == True:
  print("Processing test data")
  if number_of_layers == 3:
    x_test, m_test, n_test = mf.readintestcsv(colour);
    p_test, h_test = mf.predict3(theta1, theta2, x_test);
    np.set_printoptions(suppress=True)
    np.savetxt("mytest_3layers.csv", h_test, delimiter=",", fmt='%.17f')
    print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", lambda: " + str(lambda_reg) + ", colour: " + str(colour))
  elif number_of_layers == 4:
    x_test, m_test, n_test = mf.readintestcsv(colour);
    p_test, h_test = mf.predict4(theta1, theta2, theta3, x_test);
    np.set_printoptions(suppress=True)
    np.savetxt("mytest_4layers.csv", h_test, delimiter=",", fmt='%.17f')
    print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", lambda: " + str(lambda_reg) + ", colour: " + str(colour))
  elif number_of_layers == 5:
    x_test, m_test, n_test = mf.readintestcsv(colour);
    p_test, h_test = mf.predict5(theta1, theta2, theta3, theta4, x_test);
    np.set_printoptions(suppress=True)
    np.savetxt("l3mytest_5layers.csv", h_test, delimiter=",", fmt='%.8f')
    print("Using Hidden_layer_size: " + str(hidden_layer_size1) + ", " + str(hidden_layer_size2) + ", " + str(hidden_layer_size3) + ", lambda: " + str(lambda_reg) + ", colour: " + str(colour))
  else:
    print("Number of layers must be 3, 4 or 5")
#-----------------END BODY-----------------

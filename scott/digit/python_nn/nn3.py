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
def nncostfunction(ltheta_ravel, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg):
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


#-----------------BEGIN FUNCTION 5-----------------
def predict(ltheta1, ltheta2, x):
	m, n = np.shape(x)
	p = np.zeros(m)

	h1 = sigmoid((np.hstack((np.ones((m,1)),x.astype(float)))).dot(ltheta1.T))
	h2 = sigmoid((np.hstack((np.ones((m,1)),h1))).dot(ltheta2.T))

	for i in range(0,np.shape(h2)[0]):
		p[i] = np.argmax(h2[i,:])
	
	return p
#-----------------END FUNCTION 5-----------------


#-----------------BEGIN FUNCTION 7-----------------
def debuginitialweights(layerasize, layerbsize):
	np.random.seed(seed=2)
	w = np.random.rand(layerasize,1+layerbsize)/10.0
	np.random.seed(seed=None)
	return w
#-----------------END FUNCTION 7-----------------


#-----------------BEGIN FUNCTION 8-----------------
def smallNN(llambda_reg):
	#Some initial variables
	llambda_reg = 0
	linput_layer_size = 3;
	lhidden_layer_size = 5;
	lnum_labels = 3;
	lm = 5;
	ly = np.zeros(lm)

	# We generate some 'random' test data
	ltheta1 = debuginitialweights(lhidden_layer_size, linput_layer_size);
	ltheta2 = debuginitialweights(lnum_labels, lhidden_layer_size);
	#Reusing debugInitializeWeights to generate X
	lx = debuginitialweights(lm, linput_layer_size-1);
	np.random.seed(seed=1)
	for i in range(0,lm):
		ly[i] = np.random.randint(1,lnum_labels) #1 + mod(1:m, lnum_labels)';
	np.random.seed(seed=None)

	ltheta_ravel = np.hstack((np.ravel(ltheta1), np.ravel(ltheta2)))
	
	cost, grad = nncostfunction(ltheta_ravel, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg);

	numgrad = np.zeros(len(ltheta_ravel))
	perturb = np.zeros(len(ltheta_ravel))
	gradstep = 1e-4;
	for p in range(0,len(ltheta_ravel)):
		perturb[p] = gradstep
		loss1, lk1 = nncostfunction(ltheta_ravel - perturb, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg);
		loss2, lk2 = nncostfunction(ltheta_ravel + perturb, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg)
		numgrad[p] = (loss2 - loss1)/(2.0*gradstep)
		perturb[p] = 0

	ptheta1 = np.array(np.reshape(grad[:lhidden_layer_size * (linput_layer_size + 1)], (lhidden_layer_size, (linput_layer_size + 1))))
	ptheta2 = np.array(np.reshape(grad[lhidden_layer_size * (linput_layer_size + 1):], (lnum_labels, (lhidden_layer_size + 1))))
	ttheta1 = np.array(np.reshape(numgrad[:lhidden_layer_size * (linput_layer_size + 1)], (lhidden_layer_size, (linput_layer_size + 1))))
	ttheta2 = np.array(np.reshape(numgrad[lhidden_layer_size * (linput_layer_size + 1):], (lnum_labels, (lhidden_layer_size + 1))))

	showdiff = np.hstack((numgrad, grad))
	print(numgrad)
	print(grad)
	print('The above two columns you get should be very similar. (Right-Your Numerical Gradient, Bottom-Analytical Gradient)\n\n')
	ldiff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
	print('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).\n' + 'Relative Difference: ' + str(ldiff))
#-----------------END FUNCTION 8-----------------


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
def myoptimiser(optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold):

	#Setting initial to lower thresholds
	hidden_layer_size = d1_reg_lower_threshold
	lambda_reg = lambda_reg_lower_threshold
	list_of_hidden_layer_size = []
	list_of_lambda_reg = []
	list_of_predicted = []

	while hidden_layer_size < d1_reg_upper_threshold:
		while lambda_reg < lambda_reg_upper_threshold:
			print("Now running: hidden_layer_size:" + str(hidden_layer_size) + ", lambda_reg:" + str(lambda_reg))
			theta1_initial = randinitialize(input_layer_size, hidden_layer_size);
			theta2_initial = randinitialize(hidden_layer_size, num_labels);
			theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))
			fmin = scipy.optimize.minimize(fun=nncostfunction, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': optimisation_iteration, 'disp': False})
			answer = fmin.x
			theta1 = answer[0:hidden_layer_size*(input_layer_size+1)].reshape((hidden_layer_size, input_layer_size + 1))
			theta2 = answer[hidden_layer_size*(input_layer_size+1):].reshape((num_labels, hidden_layer_size + 1))

			p_op = predict(theta1, theta2, x_cv);
			correct_cv = [1 if a == b else 0 for (a, b) in zip(p_op,y_cv)]
			accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))

			list_of_hidden_layer_size.append(hidden_layer_size)
			list_of_lambda_reg.append(lambda_reg)
			list_of_predicted.append(accuracy_cv)

			lambda_reg = lambda_reg*3.0
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
		hidden_layer_size = int(hidden_layer_size*3.0)
		lambda_reg = lambda_reg_lower_threshold

	a = np.argmax(list_of_predicted)

	return(list_of_hidden_layer_size[a], list_of_lambda_reg[a])	
#-----------------END FUNCTION 10-----------------


#-----------------BEGIN BODY-----------------
print("Started running")

## Setup the parameters you will use for this exercise -- YOU WILL NEED TO SET THESE FLAGS BEFORE STARTING!!!
#############################################################################################################
input_layer_size  = 784           # 28x28 Input Images of Digits
hidden_layer_size = 25            # hidden units, unless allow_optimisation = True
num_labels = 10                   # 10 labels, from 0 to 9
number_of_layers = 3              # Gives the number of layers in nn. 3, 4, 5 are available.
iteration_number = 3000           # Number of iterations
lambda_reg = 1.0                  # Regularisation parameter, allow_optimisation = True
use_random_initialisation = True # If true, it will use random initialisation, if false, will use preset random values (FOR DEBUGGING ONLY) -- ONLY WORKS IF ALLOW_OPTIMISER = FALSE AND HIDDEN LAYER = 25
use_gradient_checking = False     # If true, will turn on gradient checking (FOR DEBUGGING/FIRST RUN ONLY)
minimisation_method = "L-BFGS-B"  # Sets minimiser method, recommended L-BFGS-B or TNC
use_minimisation_display = True   # Sets whether we display iterations
ratio_training_to_cv = 0.7        # Sets the ratio of training to cv data
use_all_training_data = False     # If True, will use all training data instead of spliting into train and CV
output_test_submission = True     # If True, will print out test data for submission
allow_optimisation = True         # If True, will try to find best hidden layers and lambda. It will ignore inputted numbers. Only work if use_all_training_data = False and use_random_initialisation = True
only_optimisation = False          # If True, will exit after optimisation
optimisation_iteration = 20       # Sets how many iterations when doing optimisation
lambda_reg_lower_threshold = 0.5  # Sets the min lambda threshold for optimisation
lambda_reg_upper_threshold = 100.0 # Sets the max lambda threshold for optimisation
d1_reg_lower_threshold = 25       # Sets the min d1 threshold for optimisation
d1_reg_upper_threshold = 3000     # Sets the max d1 threshold for optimisation


##Reading in data
#############################################################################################################
m, n, x, y, m_train, m_cv, x_train, x_cv, y_train, y_cv = readincsv(ratio_training_to_cv);

#Gradient checking
#############################################################################################################
if use_gradient_checking == True:
	print('Doing gradient checking')
	smallNN(lambda_reg);

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
		del(hidden_layer_size)
		del(lambda_reg)
		hidden_layer_size, lambda_reg = myoptimiser(optimisation_iteration, input_layer_size, num_labels, x_train, y_train, x_cv, y_cv, minimisation_method, lambda_reg_lower_threshold, lambda_reg_upper_threshold, d1_reg_lower_threshold, d1_reg_upper_threshold);

print("Using Hidden_layer_size: " + str(hidden_layer_size) + ", lambda: " + str(lambda_reg))

if only_optimisation == True:
	exit()

#Randomly initalize weights for Theta_initial
#############################################################################################################
print('Initialising weights')
if use_random_initialisation == True:
  theta1_initial = randinitialize(input_layer_size, hidden_layer_size);
  theta2_initial = randinitialize(hidden_layer_size, num_labels);
else:
  theta1_initial = np.genfromtxt('tt1.csv', delimiter=',')
  theta2_initial = np.genfromtxt('tt2.csv', delimiter=',')
theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))

#Minimize nncostfunction
#############################################################################################################
print('Doing minimisation')
if use_all_training_data == True:
	fmin = scipy.optimize.minimize(fun=nncostfunction, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})
else:
	fmin = scipy.optimize.minimize(fun=nncostfunction, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size, num_labels, x_train, y_train, lambda_reg), method=minimisation_method, jac=True, options={'maxiter': iteration_number, 'disp': use_minimisation_display})

answer = fmin.x
theta1 = answer[0:hidden_layer_size*(input_layer_size+1)].reshape((hidden_layer_size, input_layer_size + 1))
theta2 = answer[hidden_layer_size*(input_layer_size+1):].reshape((num_labels, hidden_layer_size + 1))

#Doing predictions
#############################################################################################################
print('Doing predictions')
if use_all_training_data == True:
	p = predict(theta1, theta2, x);
	correct = [1 if a == b else 0 for (a, b) in zip(p,y)]  
	accuracy = (sum(map(int, correct)) / float(len(correct)))  
	print 'training set accuracy = {0}%'.format(accuracy * 100)
else:
	p = predict(theta1, theta2, x_train);
	correct = [1 if a == b else 0 for (a, b) in zip(p,y_train)]
	accuracy = (sum(map(int, correct)) / float(len(correct)))
	print 'training set accuracy = {0}%'.format(accuracy * 100)

	p_cv = predict(theta1, theta2, x_cv);
	correct_cv = [1 if a == b else 0 for (a, b) in zip(p_cv,y_cv)]
	accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
	print 'CV set accuracy = {0}%'.format(accuracy_cv * 100)

#Processing test data
#############################################################################################################
if output_test_submission == True:
	print("Processing test data")

	x_test, m_test, n_test = readintestcsv();
	p_test = predict(theta1, theta2, x_test);
	imageid = []
	for i in range(len(p_test)):
		imageid.append(i+1)
	mysubmission = np.vstack((imageid,p_test))
	np.set_printoptions(suppress=True)
	np.savetxt("mytest.csv", mysubmission.T, delimiter=",", fmt='%.0f')
#-----------------END BODY-----------------

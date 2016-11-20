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

#-----------------BEGIN FUNCTION 4-----------------
def nncostfunction(ltheta_ravel, linput_layer_size, lhidden_layer_size, lnum_labels, lx, ly, llambda_reg):
	ltheta1 = np.array(np.reshape(ltheta_ravel[:lhidden_layer_size * (linput_layer_size + 1)], (lhidden_layer_size, (linput_layer_size + 1))))
	ltheta2 = np.array(np.reshape(ltheta_ravel[lhidden_layer_size * (linput_layer_size + 1):], (lnum_labels, (lhidden_layer_size + 1))))
	ltheta1_grad = np.zeros((np.shape(ltheta1)))
	ltheta2_grad = np.zeros((np.shape(ltheta2)))
	# print ltheta1[0:3,0:3],'theta1'
        # print ltheta2[0:3,0:3],'theta2'
        # print lx[0:3,0:3],'lx'
        # exit()
        y_matrix = []
	lm = np.shape(lx)[0]

	eye_matrix = np.eye(lnum_labels)
	for i in range(len(ly)):
		y_matrix.append(eye_matrix[int(ly[i])-1,:]) #The minus one as python is zero based
	y_matrix = np.array(y_matrix)

	a1 = np.hstack((np.ones((lm,1)), lx)).astype(float)

	'''
	myh = ltheta1.dot(a1.T)
	print(np.max(myh))
	#z2 = np.zeros((np.shape(myh)))
	#for i in range(0,np.shape(myh)[0]):
	#	for j in range(0,np.shape(myh)[1]):
	#		z2[i,j] = 1.0/(1.0+np.exp(-myh[i,j]))
	if lm > 1000:
		exit()
	'''
	z2 = sigmoid(ltheta1.dot(a1.T))
	a2 = (np.concatenate((np.ones((np.shape(z2)[1], 1)), z2.T), axis=1)).astype(float)
	a3 = sigmoid(ltheta2.dot(a2.T))
	h = a3
	J_unreg = 0
	J = 0
	J_unreg = (1/float(lm))*np.sum(\
	-np.multiply(y_matrix,np.log(h.T))\
	-np.multiply((1-y_matrix),np.log(1-h.T))\
	,axis=None)
	J = J_unreg + (llambda_reg/(2*float(lm)))*\
	(np.sum(\
	np.multiply(ltheta1[:,1:],ltheta1[:,1:])\
	,axis=None)+np.sum(\
	np.multiply(ltheta2[:,1:],ltheta2[:,1:])\
	,axis=None))

	delta3 = a3.T - y_matrix
	delta2 = np.multiply((delta3.dot(ltheta2[:,1:])), (sigmoidgradient(ltheta1.dot(a1.T))).T)
	cdelta2 = ((a2.T).dot(delta3)).T
	cdelta1 = ((a1.T).dot(delta2)).T

	ltheta1_grad = (1/float(lm))*cdelta1
	ltheta2_grad = (1/float(lm))*cdelta2

	theta1_hold = ltheta1
	theta2_hold = ltheta2
	theta1_hold[:,0] = 0;
	theta2_hold[:,0] = 0;
	ltheta1_grad = ltheta1_grad + (llambda_reg/float(lm))*theta1_hold;
	ltheta2_grad = ltheta2_grad + (llambda_reg/float(lm))*theta2_hold;
	thetagrad_ravel = np.concatenate((np.ravel(ltheta1_grad), np.ravel(ltheta2_grad)))

	return (J, thetagrad_ravel)	
#-----------------END FUNCTION 4-----------------


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
	g = np.multiply(sigmoid(lz),(1-sigmoid(lz)))

	return g
#-----------------END FUNCTION 3-----------------




#-----------------BEGIN FUNCTION 5-----------------
def predict(ltheta1, ltheta2, x):
	m, n = np.shape(x)
	p = np.zeros(m)

	h1 = sigmoid((np.hstack((np.ones((m,1)),x.astype(float)))).dot(ltheta1.T))
	h2 = sigmoid((np.hstack((np.ones((m,1)),h1))).dot(ltheta2.T))

	print("Start")
	print(np.shape(np.hstack((np.ones((m,1)),x.astype(float)))))
	print(np.shape(ltheta1.T))
	print("End")
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
		data.append(row)
data = np.array(data)
x = data[:,1:]
y = data[:,0]
x = x.astype(float)
y = y.astype(int)
for i in range(len(y)):
	if y[i] == 0:
		y[i] = 10

#Set basic parameters
m, n = np.shape(x)
lambda_reg = 1.0

#Randomly initalize weights for Theta_initial
#theta1_initial = np.genfromtxt('tt1.csv', delimiter=',')
#theta2_initial = np.genfromtxt('tt2.csv', delimiter=',')
theta1_initial = np.genfromtxt('tt1.csv', delimiter=',')
theta2_initial = np.genfromtxt('tt2.csv', delimiter=',')
theta_initial_ravel = np.concatenate((np.ravel(theta1_initial), np.ravel(theta2_initial)))

'''
print("theta1_initial")
print(theta1_initial[:,25])
print("theta2_initial")
print(theta2_initial[:,25])
'''

smallNN(lambda_reg);
J, grad = nncostfunction(theta_initial_ravel, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg);

print('Doing fminunc')
fmin = scipy.optimize.minimize(fun=nncostfunction, x0=theta_initial_ravel, args=(input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg), method='TNC', jac=True, options={'maxiter': 100, 'disp': False})
fmin
theta1 = np.array(np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
theta2 = np.array(np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))

'''
print("theta1")
print(theta1[:,25])
print("theta2")
print(theta2[:,25])
exit()
'''

p = predict(theta1, theta2, x);
for i in range(len(y)):
	if y[i] == 10:
		y[i] = 0

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
    x_test.append(row)

m_test, n_test = np.shape(x_test)

p_test = predict(theta1, theta2, x_test);
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

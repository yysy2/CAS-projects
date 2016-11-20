import csv
import numpy as np
# import pandas as pd
##### contains costfunctions for various algorithms, needed here: nnCostFunction for a feed-forward neural network with one hidden layer
import costfunction as cf
# contains numerical gradient checker for debugging purposes:
import gradient_checking as gc
# contains home-made sigmoid and sigmoidGradient function:
import logistics as lg
# random initialization for weights of a neural network depending on the number of incoming and outgoing connections:
import randinit as rd
import math
# import library for advanced optimization methods
import scipy.optimize as op
# contains function to give the actual predictions using the trained network
import predict as predict
# for plotting
import matplotlib.pyplot as plt

print('Loading data')
# apparently faster code, shorter one-liner alternative see below, here the header is also thrown out
###loading data from csv file - contains m examples of digits encoded in 784 pixels = number features.
### each pixel has a value of 0...255 associated with it 
### original training data [y, X] with y = column of labels (i.e. the actual digit associated with each of the m examples)
### X = 784 columns of m rows giving the actual pixel grayscale values
### note first row is header: title of pixel (pixelx with x = i * 28 + j where row i and column j and i,j element of [0,27]) 
data = []
with open('../data/train.csv', 'rb') as csv_file:
    has_header = csv.Sniffer().has_header(csv_file.read(1024))
    csv_file.seek(0)
    data_csv = csv.reader(csv_file, delimiter=',')
    if has_header:
       next(data_csv)
    for row in data_csv:
       data.append(row)
data = np.array(data,dtype=np.float64)
m = len(data[:,0])
n = len(data[0,:])
print 'Number of total examples and features (pixels): ',m,', ',n-1 
# Split data set into a training set and a cross validation set
m_train = math.ceil(m*8/10)
print m_train,' size training dataset'
m_cv = m - m_train
print '\n', m_cv,' size cross-validation dataset'
X = data[:,1:]
# print X.dtype
X_train = data[:m_train,1:]
# print X_train.dtype
X_cv = data[m_train:,1:]
# print X_cv.dtype
# exit()
y = data[:,0]
y = y.astype(int)
y_train = y[:m_train]
y_cv = y[m_train:]
### one-liner to load data: slower than alternative above:
# data = np.genfromtxt('../data/train.csv',delimiter=',')
# #### for number of training examples m I do actually include the header line, so that m effectively is m+1 
# #### since python indexing 0:m means essentially elements 1:m-1 this cancels out (think about it)
# m = len(data[:,0])
# n = len(data[0,:])
#### randomly shuffle rows to pick random subset of data for the training set
# data = np.random.shuffle(data_raw)
# data = np.array(data,dtype=np.float64)
# print data_raw.dtype
# print data.dtype
#define size of the cross-validation dataset = 20%
# m = math.ceil(m*8/10)
# m = math.ceil(m*8/10)
#m_cv = m - m
# print size of training set and cross-validation set
# print m, m_cv
# y = data[1:m,0]
# #y_cv = data[m:m,0]
# X = data[1:m,1:n]
#X_cv = data[m:m,1:n]
# print X.shape
# #print X_cv.shape
# print X.dtype

#### definition of the neural network with three layers: inputs, hidden layer, output layer
#### input layer size is the number of features (i.e. the number of pixels)
input_layer_size = 784
### hidden layer contains x knots
hidden_layer_size = 500
#### we have ten labels for digits 0,1,...9
num_labels = 10
### test the sigmoid function
print lg.sigmoid(np.array([1,-0.5,0,0.5,1]))
print 'this should be 0.73106   0.37754   0.50000   0.62246   0.73106'
print lg.sigmoidGradient(np.array([1,-0.5,0,0.5,1]))
print 'this should be 0.19661   0.23500   0.25000   0.23500   0.19661'

### randomly initialize weights of the neural network
initial_Theta1 = rd.nnrandInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = rd.nnrandInitializeWeights(hidden_layer_size,num_labels)
#### unroll the intial theta parameters into a vector:
initial_nn_params = np.hstack((initial_Theta1.flatten(),initial_Theta2.flatten()))

### Check gradient implementation is correct:
# gc.checkNNGradients(lamb)
#### now actually train the neural network on the training data:
#test various regulatization parameters lambda
# lamb = np.array([0,0.1,1,10,100,1000,5000,10000])
# lamb = np.array([0,0.1,1,10,100,1000])
lamb = np.array([0,0.3,1,10,100,1000])
# lamb = np.array([0,1,10])
nr_lamb = len(lamb)

###theta initial for test purposes
# theta1_initial = np.genfromtxt('../data/tt1.csv', delimiter=',')
# theta2_initial = np.genfromtxt('../data/tt2.csv', delimiter=',')

### number of iterations used in each advanced optimiziation method:
### this is super brute force # not to be recommended
# num_iterations = np.array([2,10,20,30,50,100])
num_iterations = np.array([12])
# num_iterations = np.array([2,4,6])
list_methods = np.array(['L-BFGS-B','CG','TNC','SLSQP','COBYLA'])
num_methods = len(list_methods)
nr_it = len(num_iterations)
ac_train = np.empty((num_methods,nr_it,nr_lamb))
ac_cv = np.empty((num_methods,nr_it,nr_lamb))
for i in range(0,nr_lamb):
   for j in range(0,nr_it):
      for k in range(0,num_methods):
         print '\n Begin advanced optimization with ~', num_iterations[j],' iterations using ',list_methods[k],' \n and regularization parameter lambda = ',lamb[i]
         fmin = op.minimize(fun=cf.nnCostFunction, x0=initial_nn_params, args=(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lamb[i]), method=list_methods[k],jac=True, options={'maxiter': num_iterations[j], 'disp': True})
         nn_params = fmin.x
         Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape((hidden_layer_size, input_layer_size + 1))
         Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape((num_labels, hidden_layer_size + 1))
         ##### now calculate accuracy of the network prediction on the training set:
         print ' \n Accuracy on the training set for lambda=',lamb[i],' using ',list_methods[k]
         pred_train = predict.nnpredict(Theta1, Theta2, X_train)
         correct = [1 if a == b else 0 for (a, b) in zip(pred_train,y_train)]  
         accuracy = (sum(map(int, correct)) / float(len(correct)))  
         ac_train[k,j,i] = accuracy*100
         print 'accuracy = {0}%'.format(accuracy * 100)
         #### test accuracy on the cross validation set
         print 'Accuracy on the cross validation set:'
         pred_cv = predict.nnpredict(Theta1, Theta2, X_cv)
         correct = [1 if a == b else 0 for (a, b) in zip(pred_cv,y_cv)]
         accuracy = (sum(map(int, correct)) / float(len(correct)))
         ac_cv[k,j,i] = accuracy*100
         print 'accuracy = {0}%'.format(accuracy * 100)

print 'Show figure for accuracy on the training set in dependence on the optimization method and regularization parameter lambda:'
# fig = plt.figure()
# colors = plt.cm.rainbow(np.linspace(0, 1, nr_lamb))
# for i in range(0,nr_lamb):
#    plt.plot(lamb,L_BFGS_B_train[0,:],label=str(lamb[i]),color=colors[i],linewidth=2,linestyle='-')
# plt.xlabel('Number iterations',size=16)
# plt.ylabel('Accuracy (%)',size=16)
# plt.legend(title='$\lambda$',loc=2)
# fig.savefig('L_BFGS_B_train.pdf')
# plt.show()
# plt.close()
fig = plt.figure()
colors = plt.cm.rainbow(np.linspace(0, 1, num_methods))
for i in range(0,num_methods):
   plt.plot(lamb,ac_train[i,0,:],label=list_methods[i],color=colors[i],linewidth=2,linestyle='-')
plt.xlabel('Regularization parameter $\lambda$',size=16)
plt.ylabel('Accuracy (%)',size=16)
plt.legend(title='Opt. Method',loc=2)
plt.title('Training set',size=16)
fig.savefig('train_'+list_methods[i]+'.pdf')
plt.show()
plt.close()
fig = plt.figure()
colors = plt.cm.rainbow(np.linspace(0, 1, num_methods))
for i in range(0,num_methods):
   plt.plot(lamb,ac_cv[i,0,:],label=list_methods[i],color=colors[i],linewidth=2,linestyle='-')
plt.xlabel('Regularization parameter $\lambda$',size=16)
plt.ylabel('Accuracy (%)',size=16)
plt.legend(title='Opt. Method',loc=2)
plt.title('Cross-validation set',size=16)
fig.savefig('cv_'+list_methods[i]+'.pdf')
plt.show()
plt.close()


   # fmt = '{:<8}{:<20}{}'
# print 'The following columns should be very similar: left the actual, right the predicted digits for a subsample'
# print(fmt.format('', 'Label', 'Predicted'))
# for i, (labeli, predi) in enumerate(zip(y_cv[0:100],pred_cv[0:100])):
#    print(fmt.format(i, labeli, predi))

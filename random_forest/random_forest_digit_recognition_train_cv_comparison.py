import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
##### contains costfunctions for various algorithms, needed here: nnCostFunction for a feed-forward neural network with one hidden layer
# import costfunction as cf
# contains numerical gradient checker for debugging purposes:
# import gradient_checking as gc
# contains home-made sigmoid and sigmoidGradient function:
# import logistics as lg
# random initialization for weights of a neural network depending on the number of incoming and outgoing connections:
import randinit as rd
import math
# import library for advanced optimization methods
# import scipy.optimize as op
# contains function to give the actual predictions using the trained network
import predict as predict
# for plotting
import matplotlib.pyplot as plt
# from multiprocessing import Process, Queue
# from multiprocessing import Pool
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

#options: n_estimators=10 by default = stands for the number of trees in the forest
# criterion = 'gini', is the GAIN function to optimize the greedy tree algorithm. the error/cost function
# here is 2a(1-a)
#class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
#min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, 
#n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)[source]

# job run on n_jobs CPUs (if -1 this is set to the number of cores)

# estimator_array=np.arange(10,210,10)
# estimator_array=np.arange(20,400,20)
# estimator_array=np.arange(10,60,10)
# estimator_array=np.arange(100,400,100)
estimator_array=np.array([10,50,150])
len_est = len(estimator_array)
ac_train = np.empty((len_est))
ac_cv = np.empty((len_est))
ac_train_entropy = np.empty((len_est))
ac_cv_entropy = np.empty((len_est))
ac_cv_2 = np.empty((len_est))
ac_cv_entropy_2 = np.empty((len_est))

def train_forest_entropy(n_estimators,boots=True,max_feat="auto",min_leaf=1):
   clf = RandomForestClassifier(n_estimators=n_estimators,criterion="entropy",n_jobs=6,bootstrap=boots,max_features=max_feat,min_samples_leaf = min_leaf)
   clf.fit(X_train, y_train)
   preds_cv = clf.predict(X_cv)
   preds_train = clf.predict(X_train)
   correct_train = [1 if a == b else 0 for (a, b) in zip(preds_train,y_train)]
   accuracy_train = (sum(map(int, correct_train)) / float(len(correct_train)))
   print ' \n Accuracy on the training set:'
   print 'accuracy = {0}%'.format(accuracy_train * 100)
   print 'Accuracy on the cross validation set:'
   correct_cv = [1 if a == b else 0 for (a, b) in zip(preds_cv,y_cv)]
   accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
   print 'accuracy = {0}%'.format(accuracy_cv * 100)
   return (accuracy_train,accuracy_cv)

def train_forest_gini(n_estimators,boots=True,max_feat="auto",min_leaf=1):
   clf = RandomForestClassifier(n_estimators=n_estimators,n_jobs=6,bootstrap=boots,max_features=max_feat,min_samples_leaf = min_leaf)
   clf.fit(X_train, y_train)
   preds_cv = clf.predict(X_cv)
   preds_train = clf.predict(X_train)
   correct_train = [1 if a == b else 0 for (a, b) in zip(preds_train,y_train)]
   accuracy_train = (sum(map(int, correct_train)) / float(len(correct_train)))
   print ' \n Accuracy on the training set:'
   print 'accuracy = {0}%'.format(accuracy_train * 100)
   print 'Accuracy on the cross validation set:'
   correct_cv = [1 if a == b else 0 for (a, b) in zip(preds_cv,y_cv)]
   accuracy_cv = (sum(map(int, correct_cv)) / float(len(correct_cv)))
   print 'accuracy = {0}%'.format(accuracy_cv * 100)
   return (accuracy_train,accuracy_cv)


for i in range(0,len_est):
   results = train_forest_gini(estimator_array[i],False)
   ac_train[i] = results[0]
   ac_cv[i] = results[1]
# for i in range(0,len_est):
#    results = train_forest_entropy(estimator_array[i],False)
#    ac_train_entropy[i] = results[0]
#    ac_cv_entropy[i] = results[1]
for i in range(0,len_est):
   results = train_forest_gini(estimator_array[i],False,0.15,5)
   ac_cv_2[i] = results[1]
for i in range(0,len_est):
   results = train_forest_gini(estimator_array[i],False,0.15)
   ac_cv_entropy_2[i] = results[1]

print ac_train_entropy
print ac_cv_entropy

fig = plt.figure()
colors = plt.cm.rainbow(np.linspace(0, 1, 4))
# plt.plot(estimator_array,ac_train,label='Training set Gini',linewidth=2,linestyle='-',color=colors[0])
plt.plot(estimator_array,ac_cv,label='CV set Gini 1',linewidth=2,linestyle='-',color=colors[0])
# plt.plot(estimator_array,ac_train_entropy,label='Training set Entropy',linewidth=2,linestyle='-',color=colors[2])
# plt.plot(estimator_array,ac_cv_entropy,label='CV set Entropy',linewidth=2,linestyle='-',color=colors[1])
plt.plot(estimator_array,ac_cv_2,label='CV set Gini 2',linewidth=2,linestyle='-',color=colors[2])
plt.plot(estimator_array,ac_cv_entropy_2,label='CV set Entropy Gini 3',linewidth=2,linestyle='-',color=colors[3])
plt.xlabel('Number of estimators (# of trees in the forest)',size=16)
plt.ylabel('Accuracy (recognizing digits as labelled)',size=16)
plt.legend(title='Set/method',loc=2)
fig.savefig('optimization_random_forests_digit_ex.pdf')
plt.show()
exit()

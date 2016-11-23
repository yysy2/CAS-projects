import csv
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier

run_nr = 0

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
X = data[:,1:]
y = data[:,0]
y = y.astype(int)

# use the sklearn random forest classifier
# meta estimator that fits a number of decision tree classifiers on various subsamples of the dataset and 
# uses averaging to improve the predictive accuracy and control over-fitting
# the sub-sample size is always the same as the original input sample size but the samples are drawn with replacement 
# if bootstrap=True (=default)


clf = RandomForestClassifier(n_estimators=5000,n_jobs=6,bootstrap=False,min_samples_leaf=2)
clf.fit(X, y)
preds_train = clf.predict(X)
correct_train = [1 if a == b else 0 for (a, b) in zip(preds_train,y)]
accuracy_train = (sum(map(int, correct_train)) / float(len(correct_train)))
print ' \n Accuracy on the training set:'
print 'accuracy = {0}%'.format(accuracy_train * 100)

#### now do the actual calculation on the unlabelled test dataset and read in predictions into a submittable csv file

print("Loading test data")
X_test = []
with open('../data/test.csv', 'rb') as csv_file2:
  has_header2 = csv.Sniffer().has_header(csv_file2.read(1024))
  csv_file2.seek(0)
  data_csv2 = csv.reader(csv_file2, delimiter=',')
  if has_header2:
    next(data_csv2)
  for row in data_csv2:
    X_test.append(row)

X_test = np.array(X_test,dtype=np.float64)
m = len(X_test[:,0])
n = len(X_test[0,:])

preds_test = clf.predict(X_test)
imageid = np.arange(1,m+1)
# preds_test = np.array(preds_test,dtype=np.integer)
print preds_test.shape
print imageid.shape

mysubmission = np.vstack((imageid,preds_test[:]))
print mysubmission.shape,'my submission'
np.set_printoptions(suppress=True)
np.savetxt('random_forest_'+'sub_test_'+ str(run_nr) + '.csv', mysubmission.T, delimiter=",", fmt='%i')
# np.savetxt(str(int(lamb))+'_'+str(num_iterations)+'_'+"theta_test_vec_two_layers.csv", nn_params, delimiter=",", fmt='%d')

exit()

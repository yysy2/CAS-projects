import numpy as np
import logistics as lg
def nnpredict(Theta1, Theta2, Theta3, X):
   m = len(X[:,0])
   y_pred = np.zeros((m,1))
   a1 = np.hstack((np.ones((m,1)),X))
   h1 = lg.sigmoid(a1.dot(Theta1.transpose()))
   a2 = np.hstack((np.ones((m,1)),h1))
   h2 = lg.sigmoid(a2.dot(Theta2.transpose()))
   a3 = np.hstack((np.ones((m,1)),h2))
   h3 = lg.sigmoid(a3.dot(Theta3.transpose()))
   for i in range(0,m):
      y_pred[i] = np.argmax(h3[i,:])
   return y_pred

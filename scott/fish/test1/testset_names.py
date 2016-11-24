
# coding: utf-8

# In[1]:

#-----------------BEGIN HEADERS-----------------
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
import csv
import scipy
np.set_printoptions(threshold=np.nan)
import contextlib
import pdb
import glob
from PIL import Image

@contextlib.contextmanager

def printoptions(*args, **kwargs):
  original = np.get_printoptions()
  np.set_printoptions(*args, **kwargs)
  yield
  np.set_printoptions(**original)
#-----------------END HEADERS-----------------
mynames = []
mynames.append(glob.glob("*.jpg"))
mynames = np.array(mynames)
#print(mynames) 

np.savetxt("test_names.csv", mynames, delimiter=",", fmt="%s")

'''
with open('test_names.csv', 'wb') as csvfile:
  mywriter = csv.writer(csvfile, delimiter=',')
  for i in range(len(mynames)):
    for j in range(len(mynames[i])):
			#print(mynames[i,j])
      mywriter.writerow(mynames[i,j])
      #del(mycsv)
      #im.close()
'''

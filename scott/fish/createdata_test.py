
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
from PIL import ImageOps

@contextlib.contextmanager

def printoptions(*args, **kwargs):
  original = np.get_printoptions()
  np.set_printoptions(*args, **kwargs)
  yield
  np.set_printoptions(**original)
#-----------------END HEADERS-----------------
mynames = []
rmynames1 = []

rmynames1.append(glob.glob("test1/*.jpg"))   #0
mynames1 = np.sort(rmynames1)
mynames = np.hstack((np.ravel(mynames1[0,:]), "end"))

i = 0
j = 0
with open('rem10test_l_100_60.csv', 'wb') as csvfile:
  mywriter = csv.writer(csvfile, delimiter=',')
  while mynames[i] != "end":
    print("Progress: " + str((float(i)/float(len(mynames)))*100.0) + "%, " + str(mynames[i]))
    im = Image.open(mynames[i])
    im = im.convert('L') #1 = B&W, L = grey
    im = ImageOps.autocontrast(im, cutoff=10, ignore=None)
    #print(im.size)
    im = im.resize((120,80),Image.ANTIALIAS) #300,160, 150, 80
    im = ImageOps.crop(im, border=10)
    #im.show()
    #exit()
    #'''
    mycsv = np.hstack((str(mynames[i]),np.ravel(np.array(im.getdata())).astype('str')))
    mywriter.writerow(mycsv)
    del(mycsv)
    #'''
    '''
    no_of_dup = 10
    for k in range(0,no_of_dup):
      mycsv = np.hstack((str('rot' + str(k) + '-' + mynames[i]),np.ravel(np.array(im.getdata())).astype('str')))
      mywriter.writerow(mycsv)
      del(mycsv)
      im = im.rotate(360.0/float(no_of_dup))
    '''
    im.close()
    i = i + 1


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
'''
mynames.append(glob.glob("train/train/ALB/*.jpg"))   #0
mynames.append(glob.glob("train/train/BET/*.jpg"))   #1
mynames.append(glob.glob("train/train/DOL/*.jpg"))   #2
mynames.append(glob.glob("train/train/LAG/*.jpg"))   #3
mynames.append(glob.glob("train/train/NoF/*.jpg"))   #4
mynames.append(glob.glob("train/train/OTHER/*.jpg")) #5
mynames.append(glob.glob("train/train/SHARK/*.jpg")) #6
mynames.append(glob.glob("train/train/YFT/*.jpg"))   #7
mynames = np.array(mynames)
'''
rmynames1 = []
rmynames2 = []
rmynames3 = []
rmynames4 = []
rmynames5 = []
rmynames6 = []
rmynames7 = []
rmynames8 = []

rmynames1.append(glob.glob("train/train/ALB/*.jpg"))   #0
rmynames2.append(glob.glob("train/train/BET/*.jpg"))   #1
rmynames3.append(glob.glob("train/train/DOL/*.jpg"))   #2
rmynames4.append(glob.glob("train/train/LAG/*.jpg"))   #3
rmynames5.append(glob.glob("train/train/NoF/*.jpg"))   #4
rmynames6.append(glob.glob("train/train/OTHER/*.jpg")) #5
rmynames7.append(glob.glob("train/train/SHARK/*.jpg")) #6
rmynames8.append(glob.glob("train/train/YFT/*.jpg"))   #7

mynames1 = np.sort(rmynames1)
mynames2 = np.sort(rmynames2)
mynames3 = np.sort(rmynames3)
mynames4 = np.sort(rmynames4)
mynames5 = np.sort(rmynames5)
mynames6 = np.sort(rmynames6)
mynames7 = np.sort(rmynames7)
mynames8 = np.sort(rmynames8)

#print(mynames1[0,0])
mynames = np.hstack((np.ravel(mynames1[0,:]), "next", np.ravel(mynames2[0,:]), "next", np.ravel(mynames3[0,:]), "next", np.ravel(mynames4[0,:]), "next", np.ravel(mynames5[0,:]), "next", np.ravel(mynames6[0,:]), "next", np.ravel(mynames7[0,:]), "next", np.ravel(mynames8[0,:]), "end"))

i = 0
j = 0
with open('train_l_20_12.csv', 'wb') as csvfile:
  mywriter = csv.writer(csvfile, delimiter=',')
  while mynames[i] != "end":
    if mynames[i] != "next":
      print("Progress: " + str((float(i)/float(len(mynames)))*100.0) + "%, " + str(mynames[i]))
      im = Image.open(mynames[i])
      im = im.convert('L') #1 = B&W, L = grey
      #print(im.size)
      im = im.resize((20,12),Image.ANTIALIAS) #300,160, 150, 80
      #im.show()
      #exit()
      mycsv = np.hstack((j,np.ravel(np.array(im.getdata()))))
      #print(np.shape(mycsv))
      #exit()
      mywriter.writerow(mycsv)
      del(mycsv)
      im.close()
    else:
      j = j + 1
    i = i + 1

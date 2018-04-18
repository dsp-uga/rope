
# coding: utf-8

# In[2]:

import keras
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
from medpy import metric
import numpy as np
import os
from keras import losses
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply,Flatten,Dense
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np 
import nibabel as nib
CUDA_VISIBLE_DEVICES = [0]
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])
#oasis files 1-457
import h5py
path='/home/bahaa/oasis_mri/OAS1_'


# In[3]:

import numpy as np
import cv2



import csv
# In[62]:

with open(r'name.csv', 'r') as f:
  reader=csv.reader(f)
  row_count = sum(1 for row in reader)  # fileObject is your csv.reader
print(row_count)
  
import csv

f1 = file('name.csv', 'rb')
f2 = file('test.csv', 'rb')

c1 = csv.reader(f1)
c2 = csv.reader(f2)

import random
masterlist = [row[0] for row in c1]

for hosts_row in c2:
  if hosts_row[0] not in masterlist:
    f3 = file('name.csv', 'a')
    c3 = csv.writer(f3)
    
    fields=[str(hosts_row[0]),str(random.randint(0,14999))+' '+str(random.uniform(0.1,0.9))]
    c3.writerow(fields)
'''
for i in range(row_count,117703):
  with open(r'name.csv', 'a') as f:
    fields=[str(i),'1 1']
    writer = csv.writer(f)
    writer.writerow(fields)

import glob
import os
import numpy as np 
import csv
import cv2
'''
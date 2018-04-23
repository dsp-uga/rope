
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


from keras.models import load_model
model = load_model('basic_dense_net_dsp_round2.h5')

print(model.summary())

import csv
# In[62]:
fields=['id','landmarks']
with open(r'name.csv', 'a') as f:
  writer = csv.writer(f)
  writer.writerow(fields)

import glob
import os
import numpy as np 
import csv
import cv2
a=glob.glob('/home/rdey/dsp_final/test/*.jpg')
X_test=[]
print(len(a))
#print(a[0].replace('/home/rdey/dsp_final/train/','').replace('.jpg',''))
for i in range (0,len(a)):
  if(i%1000==0):
    print(i)
  
  

  #print(('/home/rdey/dsp_final/train/'+str(a[i].replace('/home/rdey/dsp_final/train/','').strip())))
  temp_x=cv2.imread(('/home/rdey/dsp_final/test/'+str(a[i].replace('/home/rdey/dsp_final/test/','').strip())),1)
  
  temp_x=cv2.resize(temp_x,(64,64)).astype('float32')
  predicted=model.predict(np.array(temp_x).reshape((1,)+temp_x.shape))
  #print(predicted.shape)
  
  max_value=0
  max_loc=0
  for j in range(0,len(predicted[0])):
    if(predicted[0][j]>max_value):
      max_value=predicted[0][j]
      max_loc=j

  fields=[str(a[i].replace('/home/rdey/dsp_final/test/','').strip()).replace('.jpg',''),str(max_loc)+' '+str(max_value)]
  with open(r'name.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)


  
              
  #except:
  #    print('error',i)


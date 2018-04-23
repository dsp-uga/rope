# coding: utf-8

import keras
import os
from keras.applications.vgg16 import VGG16
import numpy as np
import cv2
<<<<<<< HEAD






# In[8]:






#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
 # freezing first layers:
 # Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

x_train_path ='/home/omid/teamRope/team-rope/data/train_np/'


model=vgg_conv
print(model.summary())

model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
# In[62]:
for i in range(0,121):
=======
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply,Flatten,Dense
from keras.models import Model
import os
import glob

x_train_path = '/home/omid/teamRope/team-rope/data/train_np_60k/'

# check if the model has been saved before, load it,
if os.path.isfile('vgg16.h5') :
    print("Loading from saved model ....")
    model = Model.load('vgg16.h5')
else:
    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet',  include_top=False, input_shape=(64, 64, 3))
     # freezing first layers:
     # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    model=vgg_conv

    x = Flatten(name='flatten')(model.layers[-2].output)
    x = Dense(8192, activation='relu', name='fc1')(x)
    x = Dense(8192, activation='relu', name='fc2')(x)
    x = Dense(15000, activation='softmax', name='predictions')(x)
    # x = Dense(15000, activation='softmax', name='predictions')()

    model = Model( input= model.input , output= x )

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())


for i in range(0, len( glob.glob( x_train_path + 'X_*.npy'))):
>>>>>>> 3783c12e3a10f1340d37cb45585679d8fb5d28c5
  print(i)

  X_train=np.load(( x_train_path + "X_train"+str(i)+".npy"))
  X_train = X_train.astype('float32')
<<<<<<< HEAD
  #X_train=X_train.reshape(X_train.shape+(1,))
=======

>>>>>>> 3783c12e3a10f1340d37cb45585679d8fb5d28c5
  y_train=np.load((  x_train_path + "y_train"+str(i)+".npy"))#.reshape(X_train.shape)
  y_train=keras.utils.to_categorical(y_train, 15000)

  model.fit([X_train], [y_train],
<<<<<<< HEAD
                    batch_size=64,
                    nb_epoch=4,
                          #validation_data=([X2_validate],[y_validate]),
                    shuffle=True)
                          #callbacks=[xyz],
                          #class_weight=class_weightt)
print('last')
X_train=np.load(( x_train_path + "../X_train_last.npy"))
X_train = X_train.astype('float32')
#X_train=X_train.reshape(X_train.shape+(1,))
y_train=np.load((x_train_path + "../y_train_last.npy"))#.reshape(X_train.shape)
y_train=keras.utils.to_categorical(y_train, 15000)

model.fit([X_train], [y_train],
                  batch_size=64,
                  nb_epoch=4,
                        #validation_data=([X2_validate],[y_validate]),
                  shuffle=True)
                        #callbacks=[xyz],
                        #class_weight=class_weightt)

# In[29]:



import h5py
model.save('basic_dense_net_dsp.h5')
=======
                    batch_size=180,
                    epochs=30,
                    validation_split=0.2,
                    shuffle=True
            )

  model.save(  'vgg16.h5' )


>>>>>>> 3783c12e3a10f1340d37cb45585679d8fb5d28c5


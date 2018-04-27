
import numpy as np
import os
import tensorflow as tf
import numpy as np
import cv2
import glob
import h5py
from keras.applications import ResNet50

xTrain_np =  glob.glob ("../../data/train_np_90k_224/X*.npy")
for train_np in xTrain_np:
	x_train_name = os.path.splitext(os.path.split(train_np)[1])[0]
	x_train=np.load(train_np)
	print(x_train.shape)
	model = ResNet50(weights="imagenet")
	preds = model.predict(x_train)
	np.save( '../../data/object_classification_out/train_' + x_train_name +'.npy', np.array(preds))

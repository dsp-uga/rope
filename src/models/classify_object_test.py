
import numpy as np
import os
import tensorflow as tf
import numpy as np
import cv2
import glob
import h5py
from keras.applications import ResNet50

xTest_np =  glob.glob ("../../data/test_np_224/X*.npy")
for test_np in xTest_np:
	x_test_name = os.path.splitext(os.path.split(test_np)[1])[0]
	x_test = np.load(test_np)
	print(x_test.shape)
	model = ResNet50(weights="imagenet")
	preds = model.predict(x_test)
	np.save( '../../data/object_classification_out/test_' + x_test_name +'.npy', np.array(preds))
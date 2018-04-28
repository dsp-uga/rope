import numpy as np
import os
import tensorflow as tf
import numpy as np
import cv2
import glob
import h5py
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.applications import InceptionResNetV2
from keras.applications import VGG19


class ObjectClassifier:
    """
    this class will handle the object classification part of the ensanmble  
    this cvlass uses pretraind models availbale in keras packages to classfiy objects. 
    this will be used as on of the inouts to our final netwrok. 
    """

    def __init__(self, input_dir, output_dir, model='resnet'):
        self.input_dir = input_dir
        self.output_dir = output_dir

        if model == 'resnet':
            self.model = ResNet50(weights="imagenet")
        elif model == 'vgg16':
            self.model = VGG16(weights="imagenet")
        elif model == 'vgg19':
            self.model = VGG19(weights="imagenet")
        elif model == 'inception':
            self.model = InceptionResNetV2(weights="imagenet")
        else:
            self.model = None


    def classify(self):
        """
        this fucntion does the prediction for inputed images
        size is assumes for all the images to be 224 
        ( this can be set duting creation of numpy files )
        :return: 
        """
        if self.model:
            return

        xTrain_np = sorted(glob.glob( os.path.join( self.input_dir ,  "X*.npy")))
        for train_np in xTrain_np:
            x_train_name = os.path.splitext(os.path.split(train_np)[1])[0]
            x_train = np.load(train_np)
            print(x_train.shape)
            preds = self.model.predict(x_train)
            np.save(os.path.join( self.output_dir , "pred_"+ x_train_name + '.npy'), np.array(preds))


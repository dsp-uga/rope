import glob

from sklearn.cluster import KMeans
import numpy as np
from sklearn.externals import joblib
import os


class KMeansClusterer:
    """
    this class does the kmeans clustering of the images 
    """
    def __init__(self, num_classes=100, train_dir=None,   test_dir= None, output_dir=None ):
        """
        initializes the class

        :param num_classes: number of clusters
        :param train_dir: dir to look for training file
        :param test_dir: directory where test files are
        :param output_dir: directory where clustered output will be saved to
        """

        self.num_classes = num_classes
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir  = output_dir
        self.model_file_name = "models/kmeans_%d_Classes.pkl" % num_classes
        self.model = None

    def load_model(self):
        """"
        this function loads the model file if it exists
        """

        if os.path.isfile(self.model_file_name):
            self.model = joblib.load(self.model_file_name)
        else :
            print('not found~!',self.model_file_name)

    def cluster(self):
        """
        for given numpy files this funciton clusteers and saves the records
        """

        if self.model is None:
            self.load_model()

        file_list = glob.glob(os.path.join(self.train_dir ,"X_*.npy"))

        for file in file_list :
            temp = np.load(file)
            temp = temp.reshape(temp.shape[0], 64*64*3)
            ret  = self.model.predict(temp)

            np.save( os.path.join(self.output_dir,"train_"+ os.path.basename(file)), ret)

        file_list = glob.glob(os.path.join(self.test_dir, "X_*.npy"))

        for file in file_list:
            temp = np.load(file)
            temp = temp.reshape(temp.shape[0], 64 * 64 * 3)
            ret = self.model.predict(temp)

            np.save(os.path.join(self.output_dir,"test_"+ os.path.basename(file)), ret)

    def train(self):
        """
        this funciton trains the kmeans model
        :return:  returns the model
        """
        file_list = glob.glob(os.path.join(self.train_dir, "X_tra*.npy"))
        ctr =0
        lst  = []
        for file in file_list:
            temp = np.load(file)
            temp = temp.reshape(temp.shape[0], 64 * 64 * 3)
            lst.append(temp)

            ctr += 1
            if ctr == 4:
                break

        lst= np.vstack(lst)

        self.model =  KMeans(n_clusters=self.num_classes, random_state=0)
        self.model.fir(lst)
        joblib.dump(self.model, self.model_file_name)

        return self.model

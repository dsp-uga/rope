from sklearn.cluster import KMeans
import numpy as np
from sklearn.externals import joblib


x_train_path ='/home/omid/teamRope/team-rope/data/train_np_90k/'

X_train = np.load((x_train_path + "X_train1.npy"))

X_train = X_train.reshape( X_train.shape[0], 64*64*3 )

# y_train = np.load((x_train_path + "y_train" + str(i) + ".npy"))  # .reshape(X_train.shape)

kmeans = KMeans(n_clusters=100, random_state=0)

kmeans.fit(X_train)

joblib.dump(kmeans, 'kmeans_100_Classes.pkl')
# print(kmeans.labels_)

# kmeans.predict([[0, 0], [4, 4]])
# print(kmeans.cluster_centers_)

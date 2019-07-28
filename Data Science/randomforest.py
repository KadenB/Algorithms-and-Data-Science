
import numpy as np
import matplotlib as pl
import numpy.matlib
import matplotlib.pyplot as plt
import math
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier

# Random Forest
#  X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # normalization
crop_data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter= ',') 
train_labels = crop_data_train[:,-1]
test = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')
test_labels = test[:,-1]

def randomF(X_train,Y_train,X_test,Y_test):
    RF = RandomForestClassifier(50)

    RF.fit(crop_data_train,train_labels)

    score = RF.score(test,test_labels)
    print(score*100)
    return score*100

randomF(crop_data_train,train_labels,test,test_labels)
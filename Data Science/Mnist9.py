# Exercise 9 and 10:
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
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

from sklearn.model_selection import train_test_split
mnist_data = np.loadtxt('MNIST_179_digits.txt')
mnist_lables = np.loadtxt('MNIST_179_labels.txt')



def get_train_test():
    X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_lables, test_size=0.10,shuffle =True)
    return X_train, X_test,y_train,y_test


def get_clusters(show_plot = False):
     X_train, X_test, y_train, y_test = get_train_test()
     kmeans = KMeans(n_clusters=3,algorithm='full').fit(X_train)
     cluster_centers = kmeans.cluster_centers_
     c1 = cluster_centers[0].reshape(28,28)
     c2 = cluster_centers[1].reshape(28,28)
     c3 = cluster_centers[2].reshape(28,28)
     
     f, ax = plt.subplots(2,3)
     ax[0,0].imshow(c1)
     ax[0,1].imshow(c2)
     ax[0,2].imshow(c3)

 
    # predicted labels
     pred = kmeans.predict(X_train)
     X_train[:,-1]= y_train
     X_train[:,0] = pred 

     class0 = X_train[np.where(X_train[:,0]==0)]
     class1 = X_train[np.where(X_train[:,0]==1)]
     class2 = X_train[np.where(X_train[:,0]==2)]

     class0_1s = len(class0[np.where(class0[:,-1]==1)])/len(class0)*100
     class0_9s = len(class0[np.where(class0[:,-1]==9)])/len(class0)*100
     class0_7s = len(class0[np.where(class0[:,-1]==7)])/len(class0)*100

     class1_1s = len(class1[np.where(class1[:,-1]==1)])/len(class1)*100
     class1_9s = len(class1[np.where(class1[:,-1]==9)])/len(class1)*100
     class1_7s = len(class1[np.where(class1[:,-1]==7)])/len(class1)*100

     class2_1s = len(class2[np.where(class2[:,-1]==1)])/len(class2)*100
     class2_9s = len(class2[np.where(class2[:,-1]==9)])/len(class2)*100
     class2_7s = len(class2[np.where(class2[:,-1]==7)])/len(class2)*100
     
     print("The percent of 1s, 7s, and 9s in class 0 are " ,class0_1s,class0_7s,class0_9s)
     print("The percent of 1s, 7s, and 9s in class 1 are " ,class1_1s,class1_7s,class1_9s)
     print("The percent of 1s, 7s, and 9s in class 2 are " ,class2_1s,class2_7s,class2_9s)


     if show_plot == True:
         return plt.show()
     else:
         return 
   




def neighbors():
     X_train, X_test, y_train, y_test = get_train_test()
    
     score = []
    
     for x in range(1,10):
         kn = KNeighborsClassifier(n_neighbors=x)
         kn.fit(X_train,y_train)
         score1 = kn.score(X_test,y_test)
         score.append([x,score1])
     
     return score




def get_max():
    scores = neighbors()
    maxi = scores[0]

    for i in scores:
        if i[1] > maxi[1]:
            maxi = i
    print(maxi)
    k = maxi[0]
    score = maxi[1]
    return k, score



def main():
    get_max()
    get_clusters(show_plot=True)

main()
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import random
from matplotlib.colors import ListedColormap
w_train = np.loadtxt('redwine_training.txt')
w_test = np.loadtxt('redwine_testing.txt')

w_train_nolabel = w_train[:,:-1]
w_train_labels = w_train[:,-1]

def lin_regression(one_feature = True):
    
    if one_feature == False:
        X = w_train_nolabel
        Y = w_train_labels
        intercept =  np.ones(shape = Y.shape)[...,None]
        X = np.concatenate((intercept,X),1)
    else:
         X = w_train_nolabel[:,0]
         Y = w_train_labels
         X = np.zeros((w_train_nolabel.shape[0], 2))
         X[:,1] = w_train_nolabel[:,0]
         X[:,0] = 1
       
       
        #W = (X^t*X)-1 XT ys
    one = np.linalg.inv(np.dot(X.T,X))
    two = np.dot(one,X.T)
    Weights = np.dot(two,Y)

    Y_pred = np.dot(X,Weights)

    error = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    print("Weights are ", str(Weights),"and error is ",str(error))
    return Weights,error

lin_regression(True)
lin_regression(False)

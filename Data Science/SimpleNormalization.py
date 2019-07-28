


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing



Test = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')

Train = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')

X = Train[:,:-1]
Xtest = Test[:,:-1]

y = Train[:,-1]
Ytest =Test[:,-1]

def first():
    knn1 = KNeighborsClassifier(n_neighbors=1)

    knn1.fit(X,y)

    accTrain= accuracy_score(y,knn1.predict(X))*100
    accTest = accuracy_score(Ytest,knn1.predict(Xtest))*100

    print("Training accuracy is " + str(accTrain) + " Testing accuracy is " + str(accTest))


## Exercise 2
def c_val(train_x,train_y):
    k = [1,3,5,7,9,11]
    cv = KFold(n_splits = 5)
    performance = {}
    
    for x in k:
        knn = KNeighborsClassifier(n_neighbors=x)
        a_scores = cross_val_score(knn, train_x,train_y, cv = cv)
        #print(x)
        #print(a_scores)
        av = a_scores.mean()
        performance[x]=av
    k_best =max(performance,key = performance.get)
    #print(performance)
    #
    print("K_best is " + str(k_best))
    return k_best


    
def k_best(train_x,train_y, test_x, test_y):
    
    k_best = c_val(train_x,train_y)
    
    knn = KNeighborsClassifier(n_neighbors=k_best)
    
    knn.fit(train_x, train_y)
    
    accTest = accuracy_score(test_y,knn.predict(test_x))*100
    accTrain = accuracy_score(y,knn.predict(X))*100
    print("Training accuracy under k= "+ str(k_best) + " is " + str(accTrain) + " and Testing accuracy is " 
          + str(accTest))


def norm(x_train,y_train,test_x,test_y):
    scalar = preprocessing.StandardScaler().fit(x_train)
    XTrainN = scalar.transform(x_train)
    XTestN = scalar.transform(test_x)
    
    c_val(XTrainN,y_train)
    
    k_best(XTrainN,y_train,XTestN,test_y)
    

def main():
    print("1-NN Results:")
    first()
    print("Un-normalized data cross validation:")   
    
    c_val(X,y)
    
    print("Unormalized accuracy with k_best:")
    
    k_best(X,y,Xtest,Ytest)
    
    print("Normalized accuracy with k_best:")    
   
    norm(X,y,Xtest,Ytest)



main()
# Script to perform predition on the Iris dataset, using a self implemented version of logistic regression


import numpy as np
import matplotlib as pl
import numpy.matlib
import matplotlib.pyplot as plt
import math
import random
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Iris 2D data

iris_train = np.loadtxt('Iris2D2_train.txt')
iris_test = np.loadtxt('Iris2D2_test.txt')
labels1 = iris_train[:,-1]
labels_test2d = iris_test[:,-1]

#Iris 1D data
iris_train2 =np.loadtxt('Iris2D1_train.txt')
iris_test2 =  np.loadtxt('Iris2D1_test.txt')
labels_test1d = iris_test2[:,-1]
labels2 = iris_train2[:,-1]

#print(iris_test2.shape)


# Plot tfunction
def plot_iris(data,labels):
    plt.figure(figsize =(8,8))
    plt.scatter(data[:,0], data[:,1], c = labels)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    return plt.show()
    


# get the log function fo a given X vector and the Weights    
def get_logfunc(x,weights):
    z = np.dot(x, weights.T)
    return 1 / (1 + math.e**(-z))



# def get_gradient(theta,X,Y):
#     g = -(1/Y.size)* np.sum([(np.dot(y,x))/(1+math.e**(np.dot(np.dot(theta.T,x),y))) for x, y in zip(X,Y)])
#     return g
    

def log_regression(data,Y,num_iterations, learning_rate): 
    

    s = data[:,:-1].shape[1]
    X = np.zeros((data[:,:-1].shape[0],s+1))

    X[:,0] = 1 # add intercept at 1
    X[:,1:] = data[:,:-1]
    
    weights=np.zeros(X.shape[1])# initialize weights as one

    for i in range(num_iterations):
        h = get_logfunc(X,weights.T) # this counts as our cost
        g = np.dot(X.T, (h - Y)) / Y.size # now we get the gradient in respect to our cost
       
        weights= weights - learning_rate* g
           
    return weights



def predict(X,weights):
    pred =np.where(get_logfunc(X,weights) >= 0.5, 1, 0)
    return pred


def get_avgerr(pred,actual):
    error = np.mean(actual - pred)
    return error

def get_score(pred,Y):
    right = 0
    wrong = 0 
    for i in range(len(pred)):
        if pred[i] == Y[i]:
            right = right + 1
        else:
            wrong = wrong +1
    
    return right/(right+wrong)*100
            


def main(train,train_labels,test,test_labels,num_iter=30000,learning_rate=.001):
    
    weights = log_regression(train,train_labels,num_iter,learning_rate)

    predicted_val_train = predict(train,weights)
    predicted_val_test = predict(test,weights)

    error_train = get_avgerr(predicted_val_train,train_labels)
    error_test = get_avgerr(predicted_val_test,test_labels)

    score_train = get_score(predicted_val_train, train_labels)
    score_test = get_score(predicted_val_test,test_labels)

    print("Weights are ", weights, "Average error for the training set, and test set respectively are", error_train, error_test, "Scores for train and test sets are", score_train,score_test)

plot_iris(iris_train,labels1)
plot_iris(iris_train2,labels2)
main(iris_train,labels1,iris_test,labels_test2d)
main(iris_train2,labels2,iris_test2,labels_test1d)

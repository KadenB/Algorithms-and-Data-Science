# Implementing gradient descent from scratch to check converstion rates


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
from numpy import sin,linspace,power
from sympy import *
import math

e = 2.71828
x = Symbol('x')

def f(x):
    return e**((-x)/2)+ 10*(x**2)

y=f(x)


def get_prime(y,x,theta):
    yp = y.diff(x)
    yp2 = yp.subs(x,theta)
    return yp2
    
    

def get_gradient(y,learning_rate,num_iter, theta0 =1):
    # create our function
    y_deriv = y.diff(x) #get our derivative
    
    y_prime = y_deriv.subs(x,theta0)
    y_numeric = y.subs(x,theta0)
    first_int = -x + y * y_prime 
    
    
    
    all_thetas= [theta0]
    all_ys = [y_numeric]
    derivatives = [y_prime]
    #intercepts = [first_int]
    count =0
    
    for i in range(num_iter):      
        theta1 = theta0  - (learning_rate*get_prime(y,x,theta0))
        all_thetas.append(theta1)# get
        update_y = y.subs(x,theta1)# get all ys for thetas
        all_ys.append(update_y)
        derivatives.append(get_prime(y,x,theta0))
        count+= 1
        
        if get_prime(y,x,theta1) < .0000000001:
            print("Converged at", count)
            break
            
        
        #print(get_prime(y,x,theta0))
        theta0 = theta1
        
        
    
    plt.figure(figsize = (6,6))
    
    
        
    plt.scatter([i for i in all_thetas],[j for j in all_ys], c ='r')
    
    
    random_x= np.linspace(-2,2,20)
    random_y = [f(i) for i in random_x] 
    
    plt.plot(random_x,random_y)
  
      
        
    return plt.show(), all_ys[count]


### Unhilight each part to get the code for parts c, and d


# Part C
# may need to adjust random_x range depending on learning rate
get_gradient(y,.1,10)  
get_gradient(y,.01,10)
get_gradient(y,.001,10)
get_gradient(y,.0001,10)

# Part D
get_gradient(y,.1,60)  
get_gradient(y,.01,10000)
get_gradient(y,.001,10000)
get_gradient(y,.0001,10000)




# Script that was created as a homework assignment to implement PCA on a dataset from scratch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# In[197]:



# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues

murder_data = np.loadtxt('murderdata2d.txt').T
crop_data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter= ',').T
#print(murder_data.shape)
#print(crop_data_train.shape)


# In[281]:


def pca_analysis_murder(data):
     ### --------------------- Exercise 1A-B ---------------------------  ###
    data_shape = data.shape
    M,N = data_shape
    x,y = data
    mean_dimensions = np.mean([x,y],1)
    #center data due to imense doubt
    c_data = data - np.transpose(np.matlib.repmat(mean_dimensions,N,1))
    # find covariance
    Sigma = np.cov(c_data)
    #find PC and variance vector
    PC, Variances = np.linalg.eig(Sigma)
    d_variances = np.diagonal(Variances)
    d_variances.sort
    s1 = [np.sqrt(i) for i in PC]
    uv = [s1[i]* d_variances[i] for i in range(len(s1))]
    print("The variances for the murder data set are" + str(d_variances) + " and the unit vectors are"+ str(uv))
    plt.scatter(c_data[0,:], c_data[1,:], label ="Centered Murder Data")
    #plt.scatter(data[0,:], data[1,:])
    #plt.plot(6.935,20.57)
    plt.plot([0, np.abs(s1[0]*Variances[0,0])], np.abs([0, s1[0]*Variances[1,0]]), 'r', label = "Eigenvector")
    plt.plot([0, np.abs(s1[1]*Variances[0,1])], np.abs([0, s1[1]*Variances[1,1]]), 'r')
    plt.plot(0, 'yo', label = "Mean")
    plt.xlabel('% Unemployed')
    plt.ylabel('Murder Rate')
    plt.legend()
    plt.show()
        

pca_analysis_murder(murder_data)


# In[491]:


def pca_analysis_crop(data):
    ### --------------------- Exercise 1C ---------------------------  ###
    data_shape = data.shape
    M,N = data_shape
    mean_dimensions = np.mean(data,1)
    #center data due to imense doubt
    c_data = data - np.transpose(np.matlib.repmat(mean_dimensions,N,1))
    # find covariance
    Sigma = np.cov(c_data)
    #find PC and variance vector
    PC, Variances = np.linalg.eig(Sigma)
    d_variances = np.diagonal(Variances)
    #print(d_variances.shape)
    s1 = [np.sqrt(i) for i in PC]
    uv = [s1[i]* d_variances[i] for i in range(len(s1))]
    
    # all_pc = PC.T.dot(c_data)
    #PCnorm = PC/np.linalg.norm(PC)
    cum_PC = np.cumsum(PC/np.sum(PC))
    #print("The variances are" + str(d_variances) + " and the unit vectors are"+ str(uv))
    
     
    ### --------------------- Exercise 2 ---------------------------  ###
    
    ep = [(np.abs(PC[i]), Variances[:,i]) for i in range(len(PC))]
    #print(ep)
    ep.sort()
    ep.reverse()
   
    W = np.hstack((ep[0][1].reshape(M,1), ep[1][1].reshape(M,1)))
    new_data= c_data.T.dot(W)
    
    pca = PCA(n_components=2, svd_solver='full')
    new_sk = pca.fit_transform(c_data.T)   
    
   
    
    ### --------------------- Plots ---------------------------  ###
    
    # Exercise 1.c.1
    fig1 = plt.figure(figsize = (8,4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(PC, "r")
    #ax1.plot(Variances,"b")
    #ax1.legend()
    #ax1.set_ylim([-2,2])
    ax1.set_title("PC versus Variance")
    ax1.set_xlabel("Primary Components")
    ax1.set_ylabel("Variance")
    
    
    # Exercise 1.c.2
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    #ax2.plot(, "r",label = "PC")
    ax2.plot(cum_PC,"b")
    ax2.set_xlabel("Primary Components")
    ax2.set_ylabel("% Captured Variance")
    ax2.set_title("Cumulative PC verus Variance")
    
    
    # Exercise 2
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(new_data, 'bo')
    ax3.plot(new_sk, 'ro')
    #ax3.legend()
    ax3.set_title('Transformed Crop Train Data')
    red = mpatches.Patch(color='red', label='Sklearn Transformed Data')
    blue = mpatches.Patch(color='blue', label='My PCA Transformed Data')
    ax3.legend(handles=[red,blue])
    #ax3.plot(c_data,'go')
    plt.show()

pca_analysis_crop(crop_data_train)


# In[490]:


#Exercise 3
def clustering():
    Train = np.loadtxt('IDSWeedCropTrain.csv',delimiter = ',')
    Test = np.loadtxt('IDSWeedCropTest.csv',delimiter = ',')
    #kmeans = KMeans(algorithm= 'full')

    starting_p = np.vstack((Train[0,],Train[1,]))

    kmeans = KMeans(n_clusters=2,n_init=1, init= starting_p,algorithm='full').fit(Train)

    cluster_centers = kmeans.cluster_centers_

    print(cluster_centers)
    
clustering()


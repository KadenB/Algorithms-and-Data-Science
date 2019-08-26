
# script to plot diatom formation using PCA and visualization
from __future__ import division
#get_ipython().magic('matplotlib inline')
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib
import math
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')



diatoms = np.loadtxt('diatoms.txt')

def all_diatoms(show_all = False):
    all_x =[diatoms[i,: : 2] for i in range(len(diatoms))]
    all_y =[diatoms[i,1: : 2] for i in range(len(diatoms))]

    all_x = [np.append(all_x[i],all_x[i][0]) for i in range(len(all_x))]
    all_y =[np.append(all_y[i],all_y[i][0])for i in range(len(all_y))]
    
    if show_all == False:
        plt.plot(all_x[3],all_y[3])
        plt.axis('equal')
        plt.show()
    else:
        for cell in range(len(all_x)):
            plt.plot(all_x[cell],all_y[cell])
        plt.show()

all_diatoms(show_all= False)


# In[237]:


#Exercise 2
def first_component():
    
    m_diatoms = np.mean(diatoms, axis =0)
    #print(m_diatoms)
    Sigma = np.cov(diatoms.T)
    #find PC and variance vector
    evals,evecs = np.linalg.eig(Sigma)
    ep = [(evals[i], evecs[:,i]) for i in range(len(evals))]
    ep.sort(key = lambda x: x[0], reverse = True)


    three_pc = ep[0:3]


    blues = plt.get_cmap('Blues')

    #print(three_pc[0][1])

    cell1 = m_diatoms - (2*np.sqrt(three_pc[0][0])*three_pc[0][1])
    cell2 = m_diatoms - (1*np.sqrt(three_pc[0][0])*three_pc[0][1])
    cell3 = m_diatoms + (1*np.sqrt(three_pc[0][0])*three_pc[0][1])
    cell4 = m_diatoms + (2*np.sqrt(three_pc[0][0])*three_pc[0][1])

    cell1 = np.append(cell1,cell1[0:2])
    cell2 = np.append(cell2,cell2[0:2])
    cell3 = np.append(cell3,cell3[0:2])
    cell4 = np.append(cell4,cell4[0:2])

    m_diatoms = np.append(m_diatoms, m_diatoms[0:2])

    #plt.plot(m_diatoms[::2], m_diatoms[1: : 2])
    plt.axis('equal')
    c1 = plt.scatter(cell1[::2], cell1[1::2],c = cell1[::2],cmap= blues )
    c2 = plt.scatter(cell2[::2], cell2[1::2], c = cell2[::2], cmap = blues)
    c3 = plt.scatter(cell3[::2], cell3[1::2], c = cell3[::2], cmap = blues)
    c4 = plt.scatter(cell4[::2], cell4[1::2], c = cell4[::2], cmap = blues)

    plt.set_cmap(blues)
    plt.show(c1)


first_component()


# In[240]:
# Generate the first second and third components and plot them in order to visualize the general diatom shape from millions of rows of data. 

def second_component():
    Sigma = np.cov(diatoms.T)
    #find PC and variance vector
    evals,evecs = np.linalg.eig(Sigma)
    ep = [(evals[i], evecs[:,i]) for i in range(len(evals))]
    ep.sort(key = lambda x: x[0], reverse = True)
    three_pc = ep[0:3]
    m_diatoms = np.mean(diatoms, axis =0)
    blues = plt.get_cmap('Blues')

    cell1 = m_diatoms - (2*np.sqrt(three_pc[1][0])*three_pc[1][1])
    cell2 = m_diatoms - (1*np.sqrt(three_pc[1][0])*three_pc[1][1])
    cell3 = m_diatoms + (1*np.sqrt(three_pc[1][0])*three_pc[1][1])
    cell4 = m_diatoms + (2*np.sqrt(three_pc[1][0])*three_pc[1][1])


    cell1 = np.append(cell1,cell1[0:2])
    cell2 = np.append(cell2,cell2[0:2])
    cell3 = np.append(cell3,cell3[0:2])
    cell4 = np.append(cell4,cell4[0:2])

    m_diatoms = np.append(m_diatoms, m_diatoms[0:2])

    #plt.plot(m_diatoms[::2], m_diatoms[1: : 2])
    plt.axis('equal')
    c1 = plt.scatter(cell1[::2], cell1[1::2], c = cell1[::2],cmap= blues, )
    c2 = plt.scatter(cell2[::2], cell2[1::2], c = cell2[::2], cmap = blues)
    c3 = plt.scatter(cell3[::2], cell3[1::2], c = cell3[::2], cmap = blues)
    c4 = plt.scatter(cell4[::2], cell4[1::2], c = cell4[::2], cmap = blues)

    plt.set_cmap(blues)
    plt.show(c1)

second_component()


def third_component():
    
    Sigma = np.cov(diatoms.T)
    #find PC and variance vector
    evals,evecs = np.linalg.eig(Sigma)
    ep = [(evals[i], evecs[:,i]) for i in range(len(evals))]
    ep.sort(key = lambda x: x[0], reverse = True)
    three_pc = ep[0:3]
    m_diatoms = np.mean(diatoms, axis =0)
    blues = plt.get_cmap('Blues')
    cell1 = m_diatoms - (2*np.sqrt(three_pc[2][0])*three_pc[1][1])
    cell2 = m_diatoms - (1*np.sqrt(three_pc[2][0])*three_pc[2][1])
    cell3 = m_diatoms + (1*np.sqrt(three_pc[2][0])*three_pc[2][1])
    cell4 = m_diatoms + (2*np.sqrt(three_pc[2][0])*three_pc[2][1])


    cell1 = np.append(cell1,cell1[0:2])
    cell2 = np.append(cell2,cell2[0:2])
    cell3 = np.append(cell3,cell3[0:2])
    cell4 = np.append(cell4,cell4[0:2])

    m_diatoms = np.append(m_diatoms, m_diatoms[0:2])

    #plt.plot(m_diatoms[::2], m_diatoms[1: : 2])
    plt.axis('equal')
    c1 = plt.scatter(cell1[::2], cell1[1::2],c = cell1[::2],cmap= blues )
    c2 = plt.scatter(cell2[::2], cell2[1::2], c = cell2[::2], cmap = blues)
    c3 = plt.scatter(cell3[::2], cell3[1::2], c = cell3[::2], cmap = blues)
    c4 = plt.scatter(cell4[::2], cell4[1::2], c = cell4[::2], cmap = blues)

    plt.set_cmap(blues)
    plt.show()

third_component()


# In[266]:


# Ex 4
def knn_projection():
    crop_data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter= ',')

    pca = PCA(n_components=2, svd_solver='full')
    whole = pca.fit_transform(crop_data_train)

    labels = crop_data_train[:,-1]

    plt.scatter(whole[:,0], whole[:,1], c = labels)

    starting_p = np.vstack((whole[0,],whole[1,]))

    kmeans = KMeans(n_clusters=2,n_init=1, init= starting_p,algorithm='full').fit(whole)

    cluster_centers = kmeans.cluster_centers_

    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='x', s=300, linewidths=10,
                color='r', zorder=10 )

    dark = plt.get_cmap('Dark2')
    plt.set_cmap(dark)
    plt.legend()
    plt.show()
    
knn_projection()


# In[262]:


#Ex 3
def toy(hidden = False):
    toy = np.loadtxt('pca_toydata.txt')
    pca = PCA(n_components=2, svd_solver='full')
    toy_set1 = pca.fit_transform(toy[:,:-2])
    toy_set2 = pca.fit_transform(toy)
    
    if hidden == True:
        plt.scatter(toy_set1[:,0], toy_set1[:,1])
        plt.show()
    else: 
        plt.scatter(toy_set2[:,0], toy_set2[:,1])
        plt.show()

toy(hidden = True)


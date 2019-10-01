

""" This script uses the housingdata.csv to experiment with some basic machine learning algorithms and grid search methods"""

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,VotingClassifier
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import GridSearchCV
import pylab as pl
import pickle
from sklearn.model_selection import cross_val_score


# In[2]:

## Read our data and seperate it ###
 
# change to directory of the housing data
data = pd.read_csv("directory name",names= ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","Label"])


## Seperate into training and testing


# In[3]:

X = data.drop(["Label"], axis =1)
Y = data.Label
train, test, train_label, test_label = train_test_split( X, Y, test_size=0.25, random_state=35)


# In[4]:

## Letsg get a few different models just to test with multiple types of parameters per model using gridsearxh
def get_nb(train,test):
    train = train
    test = test
    clf_nb = GaussianNB()
    parameters = {'priors': [None], 'var_smoothing': [1e-09,1e-15,1e-06,1e-10]}
   
    NB_grid = GridSearchCV(clf_nb, parameters,n_jobs=2)
    
    NB_grid.fit(train,train_label.astype(np.int))
    
    Grid_label = NB_grid.predict(test)
    #re_grid = accuracy_score(Grid_label, test_label.astype(np.int))
    
    best = NB_grid.best_score_
    
    best_est = NB_grid.best_estimator_
    
    return best_est
 
def get_lin(train,test):
    train = train
    test = test
   
    # create dictionary of parameters for the gridsearch to iterate through
    params = {'copy_X': [True,False], 'fit_intercept': [True,False], 'n_jobs': [None], 'normalize': [True,False]}

    # Initialize LinReg model
    clf_lin = LinearRegression()
    clf_lin.fit(train, train_label)
    
    # Get accuracy of model before gridsearch
    pre_lin_label = clf_lin.predict(test)
    #re2 = accuracy_score(pre_lin_label, test_label)
    
    # Create gridsearch for best paremeters
    LG_grid = GridSearchCV(clf_lin, params,n_jobs=2)
    LG_grid.fit(train,train_label)
    
    #Now use the best model to get a prediction - this was not used in our final 
    #prediction but simply used to test
    Grid_label = LG_grid.predict(test)
    #re_grid = accuracy_score(Grid_label, test_label)
    
    # Best prediction score from the grid search
    best = LG_grid.best_score_
    #Best estimator from the gridsearch - This is what is used in the get_best method
    best_est = LG_grid.best_estimator_ 
    
    return best_est


def get_randomF(train,test):
    train = train
    test = test
    clf_rf = RandomForestRegressor(random_state=0)
    
    params = {
    'criterion': ['mse','mae'],
    'max_features': ['auto',"sqrt","log2"],
    'min_samples_split': [2,5,10],
    }
    
    RF_grid = GridSearchCV(clf_rf, params,n_jobs=2)
    
    RF_grid.fit(train,train_label)
    
    Grid_label = RF_grid.predict(test)
    #print(Grid_label)
    #re_grid = accuracy_score(Grid_label, test_label)
    
    best = RF_grid.best_score_
    best_est = RF_grid.best_estimator_
    
    return best_est

def get_SVC(train,test):
    train = train
    test = test
    clf_sc = SVR()
    
    params = {'C': [.005,.0005,.00005],
            'kernel': ['linear', 'rbf']}
    
    SC_grid = GridSearchCV(clf_sc, params,n_jobs=2)
    
    SC_grid.fit(train,train_label)
    
    Grid_label = SC_grid.predict(test)
    #print(Grid_label)
    #re_grid = accuracy_score(Grid_label, test_label)
    
    best = SC_grid.best_score_
    best_est = SC_grid.best_estimator_
    
    return best_est  

 


# In[5]:

# Set the best classifiers so we dont have to call them everytime 

def get_best(train,test):
    
    #Method 1 Naive Bayes
    NB = get_nb(train,test) 
    # the best model
    
    #method 2 Linear Regression
    Log = get_lin(train,test)
   # the best model
    
    #method Random Forrest
    RF = get_randomF(train,test)
    
    SR = get_SVC(train,test)
 
    
    return [NB,Log,RF,SR]
    


# In[6]:

## Get our Estimators ###

all_models = get_best(train,test)


# In[9]:

### Save the model ###

pickle.dump(all_models,open( "GridSearchModels.p", "wb" ))


# In[70]:

## Get predictions of all our models we chose on the Test set ### 
NB,LR,RF,SVR =  all_models

train_label = train_label.astype("float")
test_label =  test_label.astype("float")
#fit all models
#NB.fit(train,train_label)
LR.fit(train,train_label)
RF.fit(train,train_label)
SVR.fit(train,train_label)
#NB.fit(train,train_label)


LR_score = LR.score(test,test_label)
RF_score = RF.score(test,test_label)
SVR_score = SVR.score(test,test_label)


LR_pred =LR.predict(test)
RF_pred =RF.predict(test)
SVR_pred =SVR.predict(test)

all_scores= [LR_score,SVR_score,RF_score]
print("The LR, RF, and SVR scores are as follows %s" % LR_score,RF_score,SVR_score)


# In[74]:

### Get predictions based on cross-validation 5 fold ###

NB,LR,RF,SVR =  all_models
def get_crossval(model,folds,partition = "all"):
    if partition == "train":
        cv4 = cross_val_score(model,train,train_label,cv=folds)
        cv4 = np.array(cv4)
        print("Max score is %s" % max(cv4))
        print("Min score is %s" %min(cv4))
        cv4 = np.mean(cv4)
    if partition == "test":
        cv4 = cross_val_score(model,test,test_label,cv=folds)
        cv4 = np.array(cv4)
        print("Max score is %s" %max(cv4))
        print("Min score is %s" %min(cv4))
        cv4 = np.mean(cv4)
    if partition == "all":
        cv4 = cross_val_score(model,X,Y.astype("float"),cv=folds)
        cv4 = np.array(cv4)
        print("Max score is %s" %max(cv4))
        print("Min score is %s" %min(cv4))
        cv4 = np.mean(cv4)
    return cv4

all_cross_accuracies = [get_crossval(SVR,7),get_crossval(LR,7),get_crossval(RF,7)]
print("Mean cross-validation accuracy is SVR,LR,RF %s" % all_cross_accuracies)


# In[ ]:




# In[ ]:




# In[87]:

### Lets do feature selection since our accuracy is  bit low (probably also due to very small data) ## 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


transformed_X = SelectKBest(f_regression, k=7).fit_transform(X, Y)

transformed_X.shape

# check our validation accuries
def get_crossval_new(model,folds):
    cv4 = cross_val_score(model,transformed_X,Y,cv=folds)
    cv4 = np.array(cv4)
    print("Max score is %s" % max(cv4))
    print("Min score is %s" %min(cv4))
    cv4 = np.mean(cv4)
    
    return(cv4)

scores_1 = [get_crossval_new(RF,7),get_crossval_new(LR,7),get_crossval_new(SVR,7)]
print("Scores in order RF, LR, SVR %s" % scores_1)


# In[ ]:

## Possible next steps - Normalization/standardization - other feature selection methods


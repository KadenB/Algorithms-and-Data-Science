### An experiment to evaluate the perforamnce of two hashing functions #### 
import fileinput
import random
import matplotlib.pyplot as plt

class HashExperiment:
    def __init__(self):
        self.size = 0

    def chained1(self,x,n,T):
        
        hx = 800062439*x%n
        if hx <0 :
            hx += n
        if  x not in T[hx]:
            T[hx].append(x)
        return T
    
    def chained2(self,x,n,T2):
        h1 = 800062439*x%n
        if h1 < 0:
            h1 += n
        
        h2 = 677699075 *x%n
        if h2 < 0:
            h2 += n
        if x not in T2[h1] and x not in T2[h2]:
            if len(T2[h1]) <= len(T2[h2]):
                T2[h1].append(x)
                #return T2[h1]
            else:
                T2[h2].append(x)
        return T2
  
### Determine the biggest bucket size of each hash function ###   
    def get_biggest_bucket(self,trial_list):
        n = len(trial_list)
        T =[[] for i in range(n)]
        T2 =[[] for i in range(n)]
        for x in trial_list:
            hash1 = self.chained1(x,n,T)
            hash2 = self.chained2(x,n,T2)


        #print("Hash1",hash1)
        #print("Hash2",hash2)
        hash1_max = max([len(i) for i in hash1])
        hash2_max = max([len(i) for i in hash2])
            
        return hash1_max,hash2_max
 

    def run_experiment_n(self,numtrials_list =[11,121,1331,14641,161051] ):
        # give input as a list of the sizes of an ordered list of ints ex [10,100,10000,100000]
        x_axis = numtrials_list
        y_axis_h1 = [] # store the max buckets size h1
        y_axis_h2 = [] # store the max bucket size h2
        
        for i in numtrials_list:
            imput = [x for x in range(i)] # input for the trial
            timeh1,timeh2 = self.get_biggest_bucket(imput)
            y_axis_h1.append(timeh1)
            y_axis_h2.append(timeh2)
            
        plt.plot(x_axis,y_axis_h1,label = "Hash1")
        plt.plot(x_axis,y_axis_h2, label = "Hash2")
        
        plt.legend()
        #print(y_axis_h1)
        print("Bucketsizes H1", y_axis_h1)
        print("Bucketsizes H2", y_axis_h2)
        return plt.show()
 

    def run_experiment_random(self,numtrials_list =[11,121,1331,14641] ): #161051
        # give input as a list of the sizes of an ordered list of ints ex [10,100,10000,100000]
        x_axis = numtrials_list
        y_axis_h1 = [] # store the max buckets size h1
        y_axis_h2 = [] # store the max bucket size h2
        
        for i in numtrials_list:
            imput = [random.randint(0,100000)for x in range(i)] # input for the trial
            timeh1,timeh2 = self.get_biggest_bucket(imput)
            y_axis_h1.append(timeh1)
            y_axis_h2.append(timeh2)
            
        plt.plot(x_axis,y_axis_h1,label = "Hash1")
        plt.plot(x_axis,y_axis_h2, label = "Hash2")
        
        plt.legend()
        #print(y_axis_h1)
        print("Bucketsizes H1", y_axis_h1)
        print("Bucketsizes H2", y_axis_h2)
            
        return plt.show()
        
        
##USE THIS TO CHECK HASHFUNCTION ##### 
#     def read_input(self,filename = "Hashcheck.txt"):
#         firstline = True
#         for line in fileinput.input(filename):
#             line = line.strip("\n")
#             if firstline == True:
#                 #print(line)
#                 #self.size = int(line)
                 # n = int(line)
#                 firstline = False
#             else:
#                 x = int(line)
#                 #print(x)
#                 #print(self.chained1(x,n),self.chained2(x,n))
            

    
h = HashExperiment()

sample_input_ordered = [x for x in range(0,10)]
sample_input_random = [random.randrange(0,11) for x in range(0,11)]


# In[9]:

h = HashExperiment()


# In[10]:

h.run_experiment_n()


# In[11]:

h.run_experiment_random()


# In[102]:




# In[103]:




# In[ ]:




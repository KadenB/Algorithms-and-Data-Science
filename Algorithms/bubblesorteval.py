# Comparison of two different types of bubble sort algorithms with visualization - needs some cleanup ### 

import random
import matplotlib.pyplot as plt
import time
import numpy as np

class BubbleSort:
    
    def upshift(self,A,size):
        flag = True
        while flag:
            flag = False
            for x in range(0,size-1):
                if A[x]>A[x+1]:
                    holder = A[x]
                    holder2 = A[x+1]
                    A[x]= holder2
                    A[x+1]= holder
                    flag = True
        return A
                    
    
    def alternating(self,A,size):
        flag = True
        while flag:
            for x in range(0,size-1):
                if A[x]>A[x+1]:
                    holder = A[x]
                    holder2 = A[x+1]
                    A[x]= holder2
                    A[x+1]= holder
                    flag = True
                    print(A)
            if flag:
                flag = False
                for x in range(1,size):
                    print(1)
                    if A[x]< A[x-1]:
                        holder = A[x]
                        holder2 = A[x-1]
                        A[x]= holder2
                        A[x-1]= holder
                        flag = True  
        return A
    
    def  generate_rand(self,size,sort = False):
        A = []
        if sort == False:
            for i in range(size+1):
                A.append(random.randint(0,size+1))
               ## [randint(1,5) for _ in range(10)]
        else:
            A = [ x for x in range(size+1)]
        
        return A
    
    def generate_kinv(self,K,A):
        for i in range(K+1):
            index1 = random.randint(0,len(A)-1)
            index2 = random.randint(0,len(A)-1)
            if index1 != index2:
                holder = A[index1]
                holder2 = A[index2]
                A[index1]= holder2
                A[index2]= holder
            return A
        
    def almost_sorted(self,K,size):
        if K > size:
            #print("Please choose k < size of the list")
            return"Please choose k < size of the list"
        A = self.generate_rand(size,True)
        print(A)
        almostSorted = [0]* size
        for x in range(len(A)):
            print(x)
            p1 = x-K
            p2 = x+K
            element = A[x]
            if p2>len(A)-1:
                A[x] = A[p2-x]
                A[p2-x]= element
            else:
                A[x] = A[p2]
                A[p2]= element
        return A
    
    def plot_experiment(self,K,numElements = [10,100,1000],numTimes =10,keyword = "b" ):
        if keyword == "b":
            ### Store all the times for each type of list ###
            #print("wieiejofejifeofj")
            n_s = []
            s = []
            ran = []
            
            
            #Create y axis of the average time for each type of list ###
            y_s = []
            y_r = []
            y_ns =[]

            for e in numElements:
                #print(e)
                for x in range(numTimes):
                    #print(numElements)
                    sorted_list =[i for i in range(e)]
                    randomlist = self.generate_rand(e,False)
                    nearly_sorted = self.almost_sorted(K,e)

                    # Time sorted #
                    start = time.time()
                    self.upshift(sorted_list,e)
                    end = time.time()
                    total = end - start
                    s.append(total)

                    #Time random #
                    start = time.time()
                    self.upshift(randomlist,e)
                    end = time.time()
                    total = end - start
                    ran.append(total)

                    # Time nearly sorted#
                    start = time.time()
                    self.upshift(nearly_sorted,e)
                    end = time.time()
                    total = end - start
                    n_s.append(total)
                
                y_s.append(np.mean(s))
                y_r.append(np.mean(ran))
                y_ns.append(np.mean(n_s))
                
            print("""""""Results""""""")
            
            plt.plot(numElements,y_s,label = "sorted")
            plt.plot(numElements,y_r, label = "random")
            plt.plot(numElements,y_ns, label = "nearly sorted")
            plt.legend()
            
            plt.show()
                    
                    
                
        if keyword == "a":
              ### Store all the times for each type of list ###
            print("wieiejofejifeofj")
            n_s = []
            s = []
            ran = []
            
            
            #Create y axis of the average time for each type of list ###
            y_s = []
            y_r = []
            y_ns =[]

            for e in numElements:
                
                for x in range(numTimes):
                    
                    sorted_list =[i for i in range(e)]
                    randomlist = self.generate_rand(e,False)
                    nearly_sorted = self.almost_sorted(K,e)

                    # Time sorted #
                    start = time.time()
                    self.alternating(sorted_list,e)
                    end = time.time()
                    total = end - start
                    s.append(total)

                    #Time random #
                    start = time.time()
                    self.alternating(randomlist,e)
                    end = time.time()
                    total = end - start
                    ran.append(total)

                    # Time nearly sorted#
                    start = time.time()
                    self.alternating(nearly_sorted,e)
                    end = time.time()
                    total = end - start
                    n_s.append(total)
                
                y_s.append(np.mean(s))
                y_r.append(np.mean(ran))
                y_ns.append(np.mean(n_s))
                
            print("""""""Results""""""")
            
            plt.plot(numElements,y_s,label = "sorted")
            plt.plot(numElements,y_r, label = "random")
            plt.plot(numElements,y_ns, label = "nearly sorted")
            plt.legend()
            
            plt.show()
            
#         if keyword == "both":
#             n_sb = []
#             sb = []
#             ranb = []
            
#             n_sa = []
#             sa = []
#             rana = []
            
#             y_sb = []
#             y_rb = []
#             y_nsb =[]
            
#             y_sa = []
#             y_ra = []
#             y_nsa =[]
            
            
#             for e in numElements:
#                 #print(e)
#                 for x in range(numTimes):
#                     #print(numElements)
#                     sorted_list =[i for i in range(e)]
#                     randomlist = self.generate_rand(e,False)
#                     nearly_sorted = self.almost_sorted(K,e)

#                     # Time sorted #
#                     start = time.time()
#                     self.alternating(sorted_list,e)
#                     end = time.time()
#                     total = end - start
#                     s.append(total)

#                     #Time random #
#                     start = time.time()
#                     self.alternating(randomlist,e)
#                     end = time.time()
#                     total = end - start
#                     ran.append(total)

#                     # Time nearly sorted#
#                     start = time.time()
#                     self.alternating(nearly_sorted,e)
#                     end = time.time()
#                     total = end - start
#                     n_s.append(total)
                    
                    
                    
                    
                
#                 y_s.append(np.mean(s))
# #                 y_r.append(np.mean(ran))
#                 y_ns.append(np.mean(n_s))
            
        
        
        else:
            return "Please use proper keyword: bubble, alternating, or both"
            
    
            
#l[i], l[j] = l[j], l[i]         
        
        
#     def almost_sorted(K, size):
#         A = generate_rand(size, sort = True) 
#         for x in range(0, len(A)-K, K):
#             i = randint(x, x+K-1) 
#             j = min( max(randint(i-K, i+K), len(A) ), 0) 
#             A[i], A[j] = A[j], A[i]
#             return A   
        
        
        


# In[6]:

new_sort = BubbleSort()


# In[10]:

new_sort.almost_sorted(5,11)





## Dynamic programming review in a non traditional algorithm ### 

### Rod Cutting ###

### Have a rod of X length and want to maximize the profit for each rod you cut. Have a lengths and weight table. ###


# N 
import numpy as np

prices = [1,5,8,9,10,17,17,20,24,30]
length = [1,2,3,4,5,6,7,8,9,10]

# the list of pries needs to be as big as the total rod length you want. 
def recursive_rod(p,n):
    if n ==0:
        return 0
    
    currentbest = 0
    
    for i in range(1,n+1):
        print(i)
        currentbest = max(currentbest,p[i] +recursive_rod(p,n-i))
        
    return currentbest



##Rod Cutting my way to be clear about whats happening###

##Step 1 create price list for size
price_dic ={1:1,2:5,3:8,4:9,5:10,6:17,7:17,8:20,9:24,10:30}
max_profit_per_size = {0:0,1:1}
price_array = [0,1,5,8,9,10,17,17,20,24,30] # here the index corresponds to the length of the rod, and the value corresponds to the profit
best_combination_persize={} # dictionary that should store a tuple of the combination of sizes that give the most profit


def recursive_myway(prices,size_of_rod):
    if size_of_rod == 0: # if the rod is 0 obvi the profit is 0
        return 0
    optimum_profit = 0
    for i in range(1,n+1): # for every length until we reach the total length of the rod in question
        optimum_profit = max(optimum_profit,price_array[i] + recursive_myway(price_array,n-i))# the max of the current profit not cut
    return optimum_profit


def dp_rod_bottom_up(prices,size_of_rod):
    if size_of_rod == 0:
        return 0  
    # first check if the optimal profit has already been calculated for the rod length
    
    if size_of_rod in max_profit_per_size:
        return max_profit_per_size[size_of_rod]
    else:
        for i in range(1,size_of_rod+1):# for i in the range of the total rod length
            print("I is", i)
            temperary_best = []
            for k in range(1,i+1):# for subsections up until that rod length
                print("K is", k)
                print(price_array[k])
                temperary_best.append(price_array[k] + max_profit_per_size[i-k])
            print(temperary_best)
            max_profit_per_size[i] = max(temperary_best)
    
    return max_profit_per_size[size_of_rod]
        


def dp_rod_bottom_up_table(size_of_rod):
    table = np.zeros((size_of_rod+1,size_of_rod+1))
    #intialize base case
    for i in range(size_of_rod+1):
        table[1,i] = i*price_array[1]
    # now iterate through the next rows
    for x in range(2,size_of_rod+1):
        for i in range(size_of_rod+1):   
           # print("i is",i)
            #print("X is",x)
            if x > i:
                table[x,i] = table[x-1,i]
            else:
                table[x,i] = max(table[x-1,i],price_array[x] + table[x,i-x])             
                
    return table, max(table[size_of_rod])
        
        
     
    
    
    
    
    
    
    
    


# In[208]:

print(dp_rod_bottom_up_table(4))
#print(dp_rod_bottom_up(prices,6))


# In[203]:

# ### Chain Matrix Multiplication ###
### In progress ..... ###

# # d0,d1,d2,d3,d4,d5 dimensions of all the matrices
# dimensions =[4,10,3,12,20,7]

# # M[i,j] = M[i,k]+M[k+1,j] +di-1,dk,dj

# def chain_mult(num_mat,dimensions):
#     M = np.zeros((num_mat+1,num_mat+1))
    
#     for i in range(1,num_mat+1): 
#         print("I is",i)
#         for j in range(1,num_mat+1):
            
#             best=[]
#             print("J is",j)
#             if i >= j:
#                 M[i,j]= 0
#             elif j-i == 1:
#                 M[i,j] = dimensions[i-1]*dimensions[i]*dimensions[j]
#             else:
#                 for k in range(i,j):
#                     best.append(M[i,k] +M[k+1,j]+dimensions[i-1]*dimensions[k]*dimensions[j])
#                     print(best)
#                 M[i,j] = min(best)
  
    
    
#     return M


# In[206]:

#chain_mult(4,dimensions)


# In[84]:




# In[ ]:




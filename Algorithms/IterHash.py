

### 2016 Iterated Hashing Problem to evaluate hashing performance and bucket size
# ####


       
def h(x):
    a = (x*32)
    b =int(x/2)
    c = 0x7fffffff
    d = (a^b)+1
    return c &( d )
    
def hit(i,x):
    r = x
    print("r is ",r)
    for j in range(3):
#         print("X",x)
#         print("r",r)
        r = h(r) 
#         print("new r",r)
    return r
    

def create_S():
    s = set()
    for j in range(0,1024):   
        s.add(((2**(j%5)+ j*32)*16))   
    return s
  
    
def run_program(i):
    a = list(create_S())
    hashes = []
    for x in a:
        hashes.append(hit(i,x))
    return hashes  
    
def check_buckets(i,k = [0,1,2,3,4,5,6,7]):
    print("A starting")
    print("i is...", i)

    a = list(run_program(i))
    print("A has been made")
    count_per_i = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
    
    for j in a:
        print(j)
        #print(j%8)
        count_per_i[j%8] += 1
    return count_per_i

            
            
        
       
    
    
 


# In[37]:

check_buckets(0,k = [0,1,2,3,4,5,6,7])




# In[17]:


79%8


# In[31]:




# In[32]:




# In[38]:


# def h(x):
#     a = (x*32)
#     b = int(x/2)
#     c = 0x7fffffff
#     d = (a^b)+1
#     #return c &( d )
#     return 0x7fffffff & (((x*32)^(int(x/2))) + 1)

# def hit(i, x):
#     r = x
#     for j in range(i): #no hashing if i=0 see h_0(x) = x in question
#         r = h(r)
#     return r

# def create_S():
#     s = set()
#     for j in range(0, 1024):
#         s.add((2**(j%5) + j*32)*16)
#     return s

# def run_program(i):
#     a = list(create_S())
#     hashes = []
#     for x in a:
#         hashes.append(hit(i, x))
#     return hashes

# def check_buckets(i, k = [0,1,2,3,4,5,6,7]):
#     print("A starting")
#     print("i is...", i)
    
#     a = list(run_program(i))
#     print("A has been made")
#     count_per_i = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
    
#     for j in a:
#         #print(j)
#         #print(j%8)
#         count_per_i[j%8] += 1
#     return count_per_i

#parloop(k=1...1000):
#    check_bucket(k)


# In[40]:

check_buckets(0)


# In[ ]:




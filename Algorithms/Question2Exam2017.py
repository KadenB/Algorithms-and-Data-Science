
# coding: utf-8

# In[1]:



class FindTriple:
    
    def hash_new(self,imput,seed,shift):
        return (imput * seed) #>> (64- shift)) & ((2 << (shift -1))-1)
    
    def find_triple(self,S,n):
        numBuckets = int(n/16)
       
        shift1 = math.log2(n/16)
        shift = np.int64(math.log2(n/16))
        print("Shift",shift)
        print("Shift1",shift)
        seed = random.randrange(1,1002,2)
        buckets = [[]]* numBuckets
       
        
        for x in S:
            h = self.HASH(x,seed,shift)
            print(h)
            buckets[h].append(x)
            print(buckets)
        
        for i in range(len(buckets)):
            if i == len(buckets):
                break
            else:
                e = buckets[i]
                j = buckets[i+1]
                k1 = (i+i+1)%numBuckets
                bucketk1 = buckets[k1]
                print(k1)
                if self.naive_compare(e,j,bucketk1) == 1:
                    return 1
                k2 = (i+i+1+1)%numBuckets
                bucketk2 = buckets[k2]
                if self.naive_compare(e,j,bucketk2) == 1:
                    return 1
                return 0
    
    def HASH(self,x,seed,shift):
        return (x*seed) >> (64-shift)
                
    def naive_compare(self,A,B,C) :
        for a in A:
            #print("A",a)
            for  b in B:
                #print("B",b)
                for  c in C:
                    #print("A",A," B",B, " C", C)
                    if a+b == c:
                        return 1
        return 0
                
            
    
    


# In[2]:

import math
import random
import numpy as np


# In[3]:

sample = [826795513,
555252147,
517687093,
287178182,
37180218,
78646361,
993733975,
770769204,
548354435,
430283379,
174507210,
156803224,
190532456,
877576911,
863974294,
371130207,
298799569,
488437924,
515656206,
558660592,
749193048,
547773675,
14479680,
423594387,
529594304,
401362766,
164117261,
56893334,
484320498,
252698429,
853870217,
646110279]


# In[39]:

sample = [np.int64(x) for x in sample]


# In[30]:

l = [1,2,3,4,5,6,7]


# In[4]:

mycode = FindTriple()


# In[5]:

mycode.find_triple(sample,32)


# In[76]:





# In[47]:

math.log2(2)


# In[ ]:




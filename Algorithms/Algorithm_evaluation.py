
#Script for an Algoirhtms course in which several types of algorithms need to be evaluated and compared

import random
import time
import fileinput
sample_input =  [[1,2],[3,4],[1,3],[1,4]]
sample_input2 = [[1,2],[3,4],[5,6],[7,8],[7,8],[9,10]] # set of 9 intervals



## Algoirthm to find union between sets of intervals both Naive and Sweep implementation - Commonly known as a calendar problem###

def read_input_fromfile(filename): 
        firstline = True
        intervals = []
        for line in fileinput.input(filename):
            line = line.strip("\n")
            if firstline == True:
                #print(firstline)
                num_int = firstline
                firstline = False
            else:
                start,end = line.split(" ")
                intervals.append([int(start),int(end)])
        return intervals

read_input_fromfile("input1.in.txt")


# In[2]:

sorted(sample_input,key=lambda x: x[0])

# print(max(sample_input,key = lambda x:x[0])[1])


# In[3]:
# Compare the naive implementation for overlapping time intervals
def naive(filename):
    sample_input = read_input_fromfile(filename)
    start = min(sample_input,key = lambda x:x[0])[0]
    end = max(sample_input,key = lambda x:x[0])[1]
    
    interval_overlapping = []
    
    for x in range(start,end+1):
        overlapping_for_point_x = []
        for interval in sample_input:
            if interval[0]<= x & x <= interval[1]:
                overlapping_for_point_x.append(interval)
        if len(overlapping_for_point_x) >1:
            interval_overlapping.append(overlapping_for_point_x)
    if interval_overlapping != []:
        
        return interval_overlapping[0]
    else:
        return 0


# test the implementation
naive('input1.in.txt')

# Implement a sweep based approach in which endpoints need to be sorted first
def sort_endpoints(filename):
    set_of_intervals = read_input_fromfile(filename)
    points = []
    isEndpoint = set()
    isStart = set()
    set_of_intervals = sorted(set_of_intervals,key=lambda x: x[0])
    for interval in set_of_intervals:
        isStart.add(interval[0])
        isEndpoint.add(interval[1])
        points.append(interval[0])
        points.append(interval[1])
    return points, isEndpoint,isStart
    
# Implementation for sweep based interval overlap 
def sweep(set_of_intervals):
    points,endpoints,startingpoints = sort_endpoints(filename)
    points.sort()
    #print("All boundaries", points)
    intersecting_intervals = []
    #boundaries_for_s = {}
    list_for_i = []
    s=0
    for boundary in points:
        #print("Boundary is",boundary)
        previous_s = s
        #print("Previous S",previous_s)
        if boundary in startingpoints:
            s+=1
            #print("S is starting point",s)      
        if s > previous_s:
            list_for_i.append([boundary,s])
        if boundary in endpoints:
            s-= 1
            #print("S is endpoint",s)
    maximum_s = max(list_for_i, key = lambda x: x[1])
    point_to_check = maximum_s[0]
    print("point to check", point_to_check)
    
    for interval in set_of_intervals:
        print("intervel", interval)
        if  interval[0]<= point_to_check & point_to_check <= interval[1]:
            intersecting_intervals.append(interval)
            
    return  intersecting_intervals
                
    



sample_input1 = [[1,2],[1,6],[2,4],[3,9],[8,9]] # greatest num of overlapping is 3
sample_input2 =  [[1,2],[3,4],[6,7],[8,9],[10,11]] # none overlapping
sample_input3 = [[1,2],[1,6],[2,4],[3,9],[4,9]] # all overalpping


# Timer to evaluate performance of implementations
def experiment_1(input1):
    time_sweep = 0
    time_naive = 0
    
    sweep_start = time.time()
    sweep(input1)
    sweep_end = time.time()
    time_sweep = sweep_end - sweep_start
    
    naive_start = time.time()
    naive(input1)
    naive_end = time.time()
    time_naive = naive_end - naive_start
    
    return time_sweep,time_naive
    
        
    
## Results ###

result1 = experiment_1(sample_input1)
print()
result2 = experiment_1(sample_input2)
result3 = experiment_1(sample_input3)

print(result1[1]/result1[0])
print(result2[1]/result2[0])
print(result3[1]/result3[0])

print(max(result1[1],result2[1],result3[1]))
print(result1)
print(result2)
print(result3)


import random
import time
import matplotlib.pyplot as plt

## Question 2 
# List sorting eveluation method via recursive and random sort
#    ####


def sortMed(list_to_sort):
    n = len(list_to_sort)
    if n%2 >0:
        m = int((n+1)/2)
    if n%2 == 0:
        m = int(n/2)
    
    list_to_sort.sort()
    
    return list_to_sort[m]

def recursMed(list_to_sort,m,seed =None,calls=0):
    if seed == None:
        pass
    else:
        random.seed(seed)
    n = len(list_to_sort)
    
    k = random.randrange(0,n)
    print()
    pivot = list_to_sort[k]
    Al = [x for x in list_to_sort if x<pivot]
    Ar = [x for x in list_to_sort if x>pivot]
    
    if m<= len(Al):
        calls+=1
        return recursMed(Al,m)
    elif n-m < len(Ar):
        calls+=1
        return recursMed(Ar,len(Ar)-(n-m),seed,calls)
    else:
        return pivot

def RandMed(list_to_sort,seed):
    n = len(list_to_sort)
    
    if n%2 >0:
        m = int((n+1)/2)
    if n%2 == 0:
        m = int(n/2)
    return recursMed(list_to_sort,m,seed)

# Plot the results of the different implementations for sorting.

def simple_experiment(size_of_input,seed):
    # create the same input for testing either sorted or unsorted
    sorted_input = [x for x in range(size_of_input+1)]
    unsorted_input = sorted_input
    random.seed(seed)
    random.shuffle(unsorted_input)
    
    runtime_sortmed_sorted =0
    runtime_randmed_sorted =0
    
    runtime_sortmed_shuffled =0
    runtime_randmed_shuffled =0
    
    
    ## Evaluate Runtime of the SortMed  sorted ###
    start_sm_sorted = time.time()
    
    sortMed(sorted_input)
    
    end_sm_sorted = time.time()
    
    runtime_sortmed_sorted = end_sm_sorted - start_sm_sorted
    
    ## Evaluate Runtime of RandMed sorted###
    start_rm_sorted = time.time()
    
    RandMed(sorted_input,seed)
    
    end_rm_sorted = time.time()
    
    runtime_randmed_sorted = end_rm_sorted - start_rm_sorted
    
    
    ### Evaluated Runtime of SortMed Shuffled ###
    start_sm_sh = time.time()
    
    sortMed(unsorted_input)
    
    end_sm_sh = time.time()
    
    runtime_sortmed_shuffled= end_sm_sh- start_sm_sh
    
    
    ### Evaluate Runtime of RandMed Shuffled ###
    
    start_rm_sh = time.time()
    #print(start_rm_sh)
    
    RandMed(unsorted_input,seed)
    
    end_rm_sh = time.time()
    #print(end_rm_sh)
    
    runtime_randmed_shuffled = end_rm_sh - start_rm_sh
    
    
    return runtime_sortmed_sorted,runtime_sortmed_shuffled, runtime_randmed_sorted, runtime_randmed_shuffled
    
    
def plot_simple(list_of_input_sizes,seed):
    
    runtimes_sm_sorted = []
    runtimes_sm_shuffled = []
    
    runtimes_rm_sorted = []
    runtimes_rm_shuffled = []
        
        
    # plot the runtime performance in terms of the size of x
    xaxis = list_of_input_sizes
    
    
    
    for entry in list_of_input_sizes:
        values = simple_experiment(entry,seed)
        runtimes_sm_sorted.append(values[0])
        runtimes_sm_shuffled.append(values[1])
        runtimes_rm_sorted.append(values[2])
        runtimes_rm_shuffled.append(values[3])
    
    fig = plt.figure(figsize=( 15,15))
    plt.subplot(2,1,1)
    plt.plot(xaxis,runtimes_sm_sorted,label = "SortM")
    plt.plot(xaxis,runtimes_rm_sorted, label = 'RandM')
    
    plt.title('Comparison of Sorted Input Runtimes for Random/SortMed')
    plt.xlabel('Input Size')
    plt.ylabel('Time in seconds')
    plt.xticks([x for x in range(0,1000000,100000)])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1])
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(xaxis,runtimes_sm_shuffled,label = 'SortM')
    plt.plot(xaxis,runtimes_rm_shuffled, label = 'RandM')
    plt.title('Comparison of Shuffled Input Runtimes for Random/SortMed')
    plt.xlabel('Input Size')
    plt.ylabel('Time in seconds')
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1])
    plt.xticks([x for x in range(0,1000000,100000)])
    
    plt.legend()
    plt.savefig('TEST1.png')
    
    
    return ""
    #plt.show()
    
    

    

# In[60]:

plot_simple([1001,10001,100001,1000001],23)


# In[ ]:
# Second vesion of recrusive med
def recursMed(list_to_sort,m,seed =23,calls=0):
    if seed == None:
        pass
    else:
        random.seed(seed)
    n = len(list_to_sort)
    
    k = random.randrange(0,n)
    print()
    pivot = list_to_sort[k]
    Al = [x for x in list_to_sort if x<pivot]
    Ar = [x for x in list_to_sort if x>pivot]
    
    if m<= len(Al):
        calls+=1
        print("Made call to m <= Al")
        return recursMed(Al,m)
    elif n-m < len(Ar):
        calls+=1
        print("Made call to n-m < len AR ")
        return recursMed(Ar,len(Ar)-(n-m),seed,calls)
    else:
        print("Just pivot as median")
        return pivot,calls

def RandMed(list_to_sort,seed=23):
    n = len(list_to_sort)
    
    if n%2 >0:
        m = int((n+1)/2)
    if n%2 == 0:
        m = int(n/2)
    return recursMed(list_to_sort,m,seed)

# Evaluation of a simpler version of the above 2 algoirithms where k is the same each time ##


def SimpleRandMed(list_to_sort):
    n = len(list_to_sort)
    
    if n%2 >0:
        m = int((n+1)/2)
    if n%2 == 0:
        m = int(n/2)
    return SimpleRecursiveMed(list_to_sort,m,0)

# Second version where k is fixed
def SimpleRecursiveMed(list_to_sort,m,calls=0):
    n = len(list_to_sort)
    k = 0
    print()
    pivot = list_to_sort[k]
    Al = [x for x in list_to_sort if x<pivot]
    Ar = [x for x in list_to_sort if x>pivot]
    
    if m<= len(Al):
        print("Made call to m <= Al")
        calls+=1
        print("Callse",calls)
        return SimpleRecursiveMed(Al,m,calls)
    elif n-m < len(Ar):
        print("Made call to n-m < len AR ")
        
        calls+=1
        print(calls)
        return SimpleRecursiveMed(Ar,len(Ar)-(n-m),calls)
    else:
        print("Just pivot as median")
        return pivot,calls
    


# In[ ]:

def experiment2(imput):
    num_calls_Simple =0
    num_calls_RandMed = 0
    
    valueRM =RandMed(imput)
    valueSM = SimpleRandMed(imput)
    
    num_calls_Simple += valueSM[1]
    num_calls_RandMed += valueRM[1]


    return "Number of Recursive Calls Simple ", num_calls_Simple, " Number of calls Random ", num_calls_RandMed, " for input ", imput



# In[ ]:

# Experiment where the median is the first element 2.3#

input1= [1,2,3,4,5,6,7] #list sorted median in middle 
input2 = [1,2,3,5,6,7,4] # list sorted median at end
input3 = [2,1,5,3,7,6,4] # list unsorted median at end
# experiement2(input3)

SimpleRandMed(input2)



# Attempt to generate inputs for 2.3 but due to time limit had to use manual generated inputs. 


# import statistics
# def generate_inputs(size_of_input,seed):
#     input_medianfirst = None
#     input_medianlast = None
    
#     random.seed(seed)
#     general_i = [x for x in range(size_of_input+1)]
#     print("General", general_i)
    
#     random.shuffle(general_i)
    
#     median = sortMed(general_i)
#     median_position = general_i.index(median)
#     print(median_position)
    
#     input_medianfirst = general_i
#     temp = input_medianfirst[0] 
#     print("temp 1",temp)# keep track of the first value
#     input_medianfirst[0] = median # assign the median to the first value
#     print(input_medianfirst)
#     input_medianfirst[median_position] = temp
    
#     # similar approach for moving the median to the last position
    
#     input_medianlast = general_i
#     temp2 = input_medianlast[-1]  
#     print("temp2",temp2)# keep track of the first value
#     input_medianlast[-1] = median # assign the median to the first value
#     input_medianlast[median_position] = temp2
    
#     return input_medianfirst,input_medianlast, general_i, median
    





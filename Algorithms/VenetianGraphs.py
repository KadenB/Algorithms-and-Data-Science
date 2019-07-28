### Code to check if there is a Venetian cycle in a graph using Union Find algorithm ###
import fileinput
class Graph:
    def __init__(self):
        self.graph = {}
        self.numedges = int
        self.numvertices = int
        self.edges = []
        self.isconnected = False
        self.degreeseven = False
    
    def putIfAbsent(self,vertex):
        if vertex not in self.graph:
            self.graph[vertex] =[]
            
    def add_edges(self,vertex1,vertex2):
        self.putIfAbsent(vertex1); self.putIfAbsent(vertex2)
        self.graph[vertex1].append(vertex2)
        self.graph[vertex2].append(vertex1)
        self.edges.append(tuple([vertex1,vertex2])) 

    def create_degree_array(self):
        self.degrees = []
        for vertex in self.graph:
            self.degrees.append(len(self.graph[vertex]))
                
    def check_degrees(self):
        self.create_degree_array()
        for degree in self.degrees:
            if degree%2 > 0:
                self.degreeseven = False
            else :
                self.degreeseven = True
  ##  Used guidance from inclass lectures on UnionFind as well as https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/ ###     
    def find(self,x,parent):
        if parent[x] != x:
            return self.find(parent[x],parent)
        return x

    def unionfind(self,v1,v2,size,parent):
        v1root = self.find(v1,parent)
        v2root = self.find(v2,parent)     
        if size[v1root] < size[v2root]: 
            parent[v1root] = v2root    
        elif size[v1root] > size[v2root]: 
            parent[v2root] = v1root 
        else : 
            parent[v2root] = v1root 
            size[v1root] += 1
            
    def check_connectivity(self):
        edges = self.edges  
        # Initialize each vertex as a parent of itself and set size to 0
        parent = list(self.graph.keys())
        size = [0 for x in range(len(parent))]
        for edge in edges:
            x = self.find(edge[0],parent) 
            y = self.find(edge[1],parent) 
            
            if x!=y:
                self.unionfind(edge[0],edge[1],size,parent)   
        if len(set(parent)) > 1:
            #print(set(parent))
            self.isconnected = False
        else:
            #print(parent)
            self.isconnected = True 
        
    def is_Venetian(self):
        self.check_connectivity()
        self.check_degrees()
        if self.isconnected and self.degreeseven:
            return "yes"
        else:
            return "no"
    
    def create_graph_from_input(self,filename = "ExamG3.txt"):
        firstline = True
        for line in fileinput.input(filename):
            line = line.strip("\n")
            if firstline == True:
                #print(firstline)
                self.numvertices,self.numedges = line.split(" ")
                firstline = False
            else:
                u,v = line.split(" ")
                self.add_edges(int(u),int(v))

                

sampleGraph = Graph()  
sampleGraph.create_graph_from_input()
# #answer = sampleGraph.is_Venetian()
# sampleGraph.check_connectivity()


sampleGraph.check_degrees()
sampleGraph.degrees
sampleGraph.degreeseven
#print(answer)
answer = sampleGraph.is_Venetian()
print(answer)



#graph = [(0,1),(2,3),(3,4),(9,5),(4,0),(1,2),(5,6),(7,8),(8,9),(6,7)]
# #gconnected = [(0,1),(1,2),(2,0),(2,3),(4,5),(5,6)]




#print(answer)


# In[165]:




# In[35]:

#sampleGraph.graph


# In[167]:

#sampleGraph.is_Venetian()


# In[163]:




# In[40]:




# In[43]:

#sampleGraph.edges.sort()


# In[19]:

#sampleGraph.numedges


# In[20]:

#sampleGraph.numvertices


# In[ ]:




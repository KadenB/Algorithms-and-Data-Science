""" This class creats bounding boxes for the CIFAR-10 dataset for feeding it into the neural network CNN2.py"""

import keras
from keras.datasets import cifar10
from skimage.segmentation import chan_vese # method by which to segment images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import cv2
import numpy as np
from skimage.io import imread,imsave,imshow
from skimage.filters import gaussian
from tqdm import tqdm
import pickle


# In[2]:

class BoundingBox:
    def __init__(self):
        #self.cool = None
        self.cifartrain = None
        self.cifartest = None
       
    
    # ######################################
    #steps to creating object boxes        #
    """My original goal was to try to implement a YOLO method that woul take in the 
    coordinates of the bounding boxes - which would point out where the object is
    but realized that this would be a bit out of scope for this exerize. Therefore, what
    I attempted to do was alter the CIFAR dataset with pixel data from the drawn bounding
    boxes, hoping that when it its fed to the network, the network could potentially pick up
    features in relation to the boxes
    
    """
    # 1. Threshold the image - in this case we will use Chan Ves alg
    # 2. Get contours of the threshold images
    # 3. Get the largest contour - put into a method itself for simplicity
    # 4. Create the bounding box around the highest contour
    # 5. Draw the bounding box with the bounding box dimensions
   
    ########################################
    
    def get_threshold_img(self,single_image,visualize = False):
        # convert image to black and white
        black_white = cv2.cvtColor(src = single_image, code = cv2.COLOR_BGR2GRAY)
        
        # apply a filter to smooth the image - just using default values for now
        smoothed = gaussian(black_white)
        
        # apply chan-ves filter - mu is sometimes better at higher or lower values 
        #depending on quality of photo
        image_segmented = chan_vese(smoothed, mu=0.33, lambda1=1, lambda2=1, tol=1e-3, max_iter=500,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

        # return image
        image_form1 = image_segmented[0] # this is the form we will use
        image_form2 = image_segmented[1] # good for visualization
        
        threshold_image = image_form1.astype(np.uint8) # convert the image into the format CV can handle
        
        if visualize:
            plt.imshow(threshold_image,cmap="gray")
        return threshold_image
    
    def get_contours(self,thres_image,visualize = False):
        # create copy
        copy_img = thres_image.copy()
        
        contours, hierarchy = cv2.findContours(copy_img,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        num_contours = len(contours) # just for checking
        
        # for visualization if we want to examine the contours of an image
        if visualize:
            contoured_image = cv2.drawContours(copy_img, contours, -1, (0, 0, 255), 1) 
            plt.imshow(contoured_image)
            print("The number of contours are" + str(num_contours))
        
        return contours
    
    def get_max_contour(self,all_contours, verbose = False):
        # sort in ascending order by the size of the contour
        sorted_contours = sorted(all_contours, key = lambda x: x.shape[0])
        
        largest = sorted_contours[-1]
        s = largest.shape
        
        if verbose:
            print("Shape of largest contour is" + str(s))
        return largest
    
    def create_bounding_box(self,chosen_contour):
        x, y, w, h = cv2.boundingRect(chosen_contour)
        return (x,y,w,h)
    
    
    def get_image_with_bb(self,orig_img,box_coord,visualize = False):
        # returns the image edited such that a red bounding box is drawn
        # if visualize = true then we can visualize the image for testing purposes
        image_copy = orig_img.copy()
        
        image_height = orig_img.shape[0] -1
        image_width = orig_img.shape[1] -1
        
        x,y,w,h = box_coord
        
        # here we create the image with the box, we take the min between the image and the coordinates
        image_with_box = cv2.rectangle(image_copy, (x, y), (min(x+w,image_width), min(y+h,image_height)), color = (255,40, 40), thickness = 1)
                                       
        
        if visualize:
            plt.imshow(image_with_box)
                                       
        return image_with_box
                                         
        
    def boundingBox(self,image,visualize =False):
        
        threshold = self.get_threshold_img(image)
        contours = self.get_contours(threshold)
        maxcountour = self.get_max_contour(contours)
        box_coord = self.create_bounding_box(maxcountour)
        output_img = self.get_image_with_bb(image,box_coord)
        
        
        if visualize:
            plt.imshow(threshold,cmap = "gray")
            plt.imshow(output_img)
        
        return output_img
    
        
    def replace_CIFAR(self,data,folder,dataname):
        # method to replace the CIFAR dataset with the boxed images
        data_new = data
        number_inst = data.shape[0]
        count = 0
        for i in tqdm(range(number_inst)): 
            boxed_image = self.boundingBox(data[i])
            data_new[i] = boxed_image
            if count <100:
                imsave(folder+"/"+"Image"+dataname+str(count)+".png",data_new[i])
                #plt.imshow(boxed_image)
                #print("boxed image shape,",boxed_image.shape)  
                count+=1
            
        return (data_new)
                
        
        


# In[3]:

# Get our CIFAR data to convert # 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#sample_image = imread("cat4.jpg")
pr = BoundingBox()




# In[ ]:

#pr.replace_CIFAR(x_train,x_test,"/Users/kadenbehan/Desktop/Cases/Sampleimg")

# replace the training and testing images seperately - this takes sometime so I have already included the pickled versions
new_train  = pr.replace_CIFAR(x_train,"/Users/kadenbehan/Desktop/Cases/Sampleimg","Train")
new_test  = pr.replace_CIFAR(x_test,"/Users/kadenbehan/Desktop/Cases/Sampleimg","Test")

print(new_train.shape)



# In[21]:

pickle.dump(new_test,open( "Cifar_boxed_test.p", "wb" ))


# In[27]:

pickle.dump(new_train,open( "Cifar_boxed_train.p", "wb" ))


# In[96]:

new_dict = pickle.load(infile)


# In[98]:




# In[99]:




# In[ ]:




# In[105]:

imsave("samp.png",img)


# In[ ]:




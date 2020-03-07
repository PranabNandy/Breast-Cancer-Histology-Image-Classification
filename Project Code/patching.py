import warnings     # to suppress all warnings
from random import shuffle  # shuffle is used to shuffle the numbers generated randomly
import numpy as np
from keras.utils import np_utils
import keras.backend as K    # to set the image dimensions wrt channels
K.set_image_data_format('channels_last')  # sets the image as (x, x, 3) i.e. the channel dimensions appear last
###
from skimage.transform import rotate  # to rotate the images

from skimage import io
from skimage.transform import resize   # to resize the images
warnings.filterwarnings('ignore')   # to ignore all warnings
def save_insitu(image,count,fullpath):  # just to save images
    io.imsave(fullpath,image)
    
def save_benign(image,count,fullpath):
    io.imsave(fullpath,image)

def save_invasive(image,count,fullpath):
    io.imsave(fullpath,image)
  
def save_normal(image,count,fullpath):
    io.imsave(fullpath,image)
     
def read_insitu():  # function to read insitu images and generate patches
    patches=[]  
    #y=1
    l=[]
    for i in range (0,72): # 72 images is basically test and train insitu images combined
        l.append(i)   
        shuffle(l)  # we shuffle the list to generate train, test and validation sets randomly
        shuffle(l)
        shuffle(l)
    count=0
    for i in range(0,72):
        num=l[i]
        image="In Situ/" +"t"+str(num)+".tif"  # image name

        im = io.imread((image))  # it returns a 3D np-array of shape (2048, 1536, 3)

        # we now will do the real patching work 

        for m in range(0,5):  # 5 is the number of times we need to move vertically (row-wise) to cover all the 1536 pixels
            for n in range(0,7):  # 7 is the number of times we need to move horizontally (column-wise) to cover all the 2048 pixels
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])  # so for the 1st time we take the pixels in the range (0,512) horizontally and vertically and append it to the patches[] array
                # 0 512 0 512
                # 0 512 256 768
                # 0 512 512 1024
                # 0 512 768 1280
                # 0 512 1024 1536
                # 0 512 1280 1792
                # 0 512 1536 2048    //so we have completely covered the first row taking strides of 256 pixels in each move
                # 256 768 0 512    // and we continue like this for the entire image

                im2=resize(patches[-1],(224,224,3),mode='constant')  # we take the last item of the patches list and we resize it from (512, 512) to (224, 224, 3) beacuse our model accepts this as the input image size
                for o in range(0,4):  # 4 is for the four degree of rotations as we rotate by 90 each time
                    rotated=rotate(im2,o*90)		# it returns the array for the rotated image
                    
                    if(i<5):
                        fullpath="Test/In Situ/"   # we put the first 5 randomly generated insitu images in our test set (in total we have 5*280 patches of 5 images in test set)
                    elif(i>=5 and i<10):
                        fullpath="Validation/In Situ/"  # we make the validation set here with 5 images (in total we have 5*280 patches of 5 images in validation set)
                    else:
                        fullpath="Train/In Situ/"   # we put the rest of 62 images in the train set (in total we have 62*280 patches of 5 images in validation set)
                    count=count+1                   # for image name
                    path=fullpath+str(count)+".png"   # Output path for the patches
                    save_insitu(rotated,count,path)   # we save the rotated patch
                    flipped=np.fliplr(rotated)        # we vertically flip the image patch
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)   # we save the flipped patch as a different patch
                  
        print(image)            
        print(path)
   
def read_Benign():
    patches=[]
    #y=1
    l=[]
    for i in range (0,78):
        l.append(i)
        shuffle(l)
        shuffle(l)
        shuffle(l)

    count=0
    for i in range(0,78):
        num=l[i]
        image="Benign/" +"t"+str(num)+".tif"

        
        im = io.imread((image))
        #print(im.shape)
        for m in range(0,5):
            for n in range(0,7):
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])
                im2=resize(patches[-1],(224,224,3),mode='constant')
                for o in range(0,4):
                    rotated=rotate(im2,o*90)
                    #RESIZED_image.append(rotated)
                    #CORRECT_labels.append(y)
                    if(i<5):
                        fullpath="Test/Benign/"
                    elif(i>=5 and i<10):
                        fullpath="Validation/Benign/"
                    else:
                        fullpath="Train/Benign/"
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(rotated,count,path)
                    flipped=np.fliplr(rotated)
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)
                    
                    #RESIZED_image.append(flipped)
                    #CORRECT_labels.append(y)
        print(image)            
        print(path)
        #return RESIZED_image,CORRECT_labels
def read_invasive():
    patches=[]
    #y=1
    l=[]
    for i in range (0,71):
        l.append(i)
        shuffle(l)
        shuffle(l)
        shuffle(l)

    count=0
    for i in range(0,71):
        num=l[i]
        image="Invasive/" +"t"+str(num)+".tif"

        im = io.imread((image))
        #print(im.shape)
        for m in range(0,5):
            for n in range(0,7):
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])
                im2=resize(patches[-1],(224,224,3),mode='constant')
                for o in range(0,4):
                    rotated=rotate(im2,o*90)
                    #RESIZED_image.append(rotated)
                    #CORRECT_labels.append(y)
                    if(i<5):
                        fullpath="Test/Invasive/"
                    elif(i>=5 and i<10):
                        fullpath="Validation/Invasive/"
                    else:
                        fullpath="Train/Invasive/"
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(rotated,count,path)
                    flipped=np.fliplr(rotated)
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)
                    
                    #RESIZED_image.append(flipped)
                    #CORRECT_labels.append(y)
        print(image)            
        print(path)
        #return RESIZED_image,CORRECT_labels
def read_normal():
    patches=[]
    #y=1
    l=[]
    for i in range (0,65):
        l.append(i)
        shuffle(l)
        shuffle(l)
        shuffle(l)

    count=0
    for i in range(0,65):
        num=l[i]
        image="Normal/" +"t"+str(num)+".tif"
        
        im = io.imread((image))
        #print(im.shape)
        for m in range(0,5):
            for n in range(0,7):
                patches.append(im[256*m:256*(m+2),256*n:256*(n+2)])
                im2=resize(patches[-1],(224,224,3),mode='constant')
                for o in range(0,4):
                    rotated=rotate(im2,o*90)
                    #RESIZED_image.append(rotated)
                    #CORRECT_labels.append(y)
                    if(i<5):
                        fullpath="Test/Normal/"
                    elif(i>=5 and i<10):
                        fullpath="Validation/Normal/"
                    else:
                        fullpath="Train/Normal/"
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(rotated,count,path)
                    flipped=np.fliplr(rotated)
                    count=count+1
                    path=fullpath+str(count)+".png"
                    save_insitu(flipped,count,path)
                    #RESIZED_image.append(flipped)
                    #CORRECT_labels.append(y)
        print(image)            
        print(path)

read_insitu()
read_Benign()
read_invasive()          
read_normal()
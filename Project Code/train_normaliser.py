from skimage import io
import stain_utils as utils
import stainNorm_Macenko
import numpy as np
import matplotlib.pyplot as plt

norm=stainNorm_Macenko.normalizer()  # normalizer() is the algorithm function for Macenko 

print("BENIGN")
for i in range(0,69):    # since we have 69 images in benign class
    path="Benign"+"/"  #input path for images
    path_norm="Benign_m_norm"+"/"   #output path to keep the normalised output
    n="t"+str(i)+".tif"   # image name
    fullpath= path+n   # path to the input image
    fullpath_norm= path_norm+n  # output path to the normalised image
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath) # for reading images as arrays
    if(i==0):
        #print(i1)
        
        norm.fit(i1) # for the first image we read, we need to fit the Mackenko normaliser() function in accordance with the first image so that we can normalise other images on this basis...norm.fit() actually 
                     # comes up with the values of mean and sd so that we can normalise the images.
        
        io.imsave((fullpath_norm),i1)  #save the first image
    else:
        i2=norm.transform(i1)  # apply the Mackenko transformation algorithm for the image
        io.imsave((fullpath_norm),i2)  # save the image

print("IN SITU")        
for i in range(0,63):
    path="In Situ"+"/"
    path_norm="In Situ_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==0):
        #print(i1)
        
        norm.fit(i1)
        
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
    #im = io.imread((fullpath))  
        io.imsave((fullpath_norm),i2)

print("INVASIVE")
for i in range(0,62):
    path="Invasive"+"/"
    path_norm="Invasive_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==0):
        #print(i1)
        
        norm.fit(i1)
        
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
    #im = io.imread((fullpath))  
        io.imsave((fullpath_norm),i2)

norm.fit("Normal/t0.tif")
print("NORMAL")
for i in range(0,56):
    path="Normal"+"/"
    path_norm=""
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==0):
        #print(i1)
        
        norm.fit(i1)
        
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
    #im = io.imread((fullpath))  
        io.imsave((fullpath_norm),i2)
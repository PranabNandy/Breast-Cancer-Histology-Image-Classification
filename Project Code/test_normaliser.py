from skimage import io
import stain_utils as utils
import stainNorm_Macenko
import numpy as np
import matplotlib.pyplot as plt

norm=stainNorm_Macenko.normalizer()

import numpy as np

print("BENIGN")
for i in range(1,10):
    path="Benign"+"/"
    path_norm="Benign_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        norm.fit(i1)
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1) 
        io.imsave((fullpath_norm),i2)

print("IN SITU")        
for i in range(1,10):
    path="In Situ"+"/"
    path_norm="In Situ_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        norm.fit(i1)       
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
        io.imsave((fullpath_norm),i2)
        
print("INVASIVE")
for i in range(1,10):
    path="Invasive"+"/"
    path_norm="Invasive_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        norm.fit(i1)
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
        io.imsave((fullpath_norm),i2)
print("NORMAL")
for i in range(1,10):
    path="Normal"+"/"
    path_norm="Normal_m_norm"+"/"
    n="t"+str(i)+".tif"
    fullpath= path+n
    fullpath_norm= path_norm+n
    print(fullpath)
    print(fullpath_norm)
    i1=utils.read_image(fullpath)
    if(i==1):
        norm.fit(i1)       
        io.imsave((fullpath_norm),i1)
    else:
        i2=norm.transform(i1)
        io.imsave((fullpath_norm),i2)
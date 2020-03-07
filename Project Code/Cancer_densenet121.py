
from skimage import io,color
from skimage.transform import resize
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense,Dropout,Concatenate, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model     # load_model is used to load the saved model
from keras.initializers import glorot_uniform  # it initialises the weights according to some distribution
from keras.optimizers import *
from matplotlib import pyplot as plt
from numpy import array
from matplotlib.pyplot import imshow
import keras.backend as K
import json
K.set_image_data_format('channels_last')
K.set_learning_phase(1)   
from keras.preprocessing.image import ImageDataGenerator    # ImageDataGenerator is used for handling bunches of images. It reads the images from folders and also labels them like as 0 for benign, 1 for insitu, 2 for invasive, 3 for normal
bn='bn_layer_'
conv='conv_layer_'
fc= 'fc_layer_'
k=32      # here k = no of channels at each layer
def save_history(history,file):
    with open(file, 'w') as f:
        json.dump(history, f)
    '''
    data = dict()
    with open('mydatafile') as f:
        data = json.load(f)
    '''
def bottleneck_composite(l,layer):   # everytime we call this function we create two layers
    # bottleneck layer
    X=l
    if type(l) is list:
        if(len(l)==1):
            X=l[0]
        else:
            X=Concatenate(axis=-1)(l)   # for channel-wise concatenation like if we have two 20x20 images to concatenate then by doing axis=-1 we want the result to be (20, 20, 2) and not (20,40)

    X = BatchNormalization(axis = 3, name = bn + str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(4*k, (1, 1), strides = (1, 1),padding='same', name = conv + str(layer), kernel_initializer = glorot_uniform(seed=0))(X)  # to limit each layer input channel size to 4*k
    X = Dropout(0.8)(X)
    # Composite layer
    layer=layer+1
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (3, 3), strides = (1, 1),padding='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    return X
    
    
layer=0    
def chexnet(classes=4,input_shape=(224,224,3)):
    X_input = Input(input_shape)   # X_input is a 4-D tensor (None, 224, 224, 3) where None would be replaced by batch size.
    layer=0
    layer=layer+1
    X = ZeroPadding2D((3, 3))(X_input)  # ZeroPadding2D is a function of the form ZeroPadding2D(( , ))(X) which pads onto the given array X
    X = BatchNormalization(axis = 3, name = bn + str(layer))(X)   # axis=3 indicates we batchnormalise channel-wise by doing (x-mean)/sd or something similar
    X = Activation('relu')(X)
    X = Conv2D(2*k, (7, 7), strides = (2, 2), name = conv + str(layer), kernel_initializer = glorot_uniform(seed=0))(X)  # 2*k is the no of filters/kernels, 7x7 is the kernel/filter size and seed=0 ensures consistency in results 
    X = Dropout(0.8)(X)    # 80% dropout
    print(X.shape)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    print(X.shape)
    #Dense Block = 1
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,5):
        layer=layer+2         # we increment layer by 2 because each time we bottleneck_composite() we created two layers
        X=bottleneck_composite(l,layer)
        l.append(X)
    print(X.shape)
    # Transition layer = 1   
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    print(X.shape)
    
    #Dense Block = 2
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,11):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    
    print(X.shape)
    # Transition layer = 2
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    print(X.shape)
    #Dense Block = 3
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,23):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    print(X.shape)
    # Transition layer = 3
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    print(X.shape)
    #Dense Block = 4
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,15):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    print(X.shape)
    layer=layer+2
    print(X.shape)
    X=  GlobalAveragePooling2D()(X)  # it works for each channel so if we have 32 channels we have 32 outputs
    print(X.shape)
   
    X = Dense(classes, activation='softmax', name=  fc  +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)  # output layer
    print(X.shape)
    model = Model(inputs = X_input, outputs = X, name="DenseNet121")   # this line builds the entire architecture pipeline
    
    return model


adam=Adam(lr=0.001)   # we start with a random learning rate
model = chexnet(classes = 4,input_shape = (224,224,3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    
#model.summary()
train_datagen = ImageDataGenerator( rescale=1./255)    
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(   # we need to mention from which directory images will come from
        'Train',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')     # categorical means one-hot encoded ytrue values i.e. it alphabetically classifies the different classes and gives them a unique number( 0 for benign, 1 for insitu, 2 for invasive, 3 for normal)
validation_generator = val_datagen.flow_from_directory(
        'Validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
test_generator=test_datagen.flow_from_directory(
        'Test',
        target_size=(224,224), 
        batch_size=32,
        class_mode='categorical')
print(train_generator.class_indices) # print out the class names along with their unique nos.
print(test_generator.class_indices)
print(validation_generator.class_indices)

history=model.fit_generator(train_generator, epochs =25,steps_per_epoch=2153,validation_data=validation_generator, validation_steps=175)  # steps_per_epoch = no of batches in the training set = 68880/32 = 2153
model.save('my_densenet25_with_dropout_new_data')
save_history(history.history,'history_densenet25_with_dropout_new_data')
    #print(history_files[i])

# Prediction phase
preds = model.evaluate_generator(train_generator, steps=2153) # preds is an array
print ("train Loss = " + str(preds[0]))  # preds[0] = loss
print ("train Accuracy = " + str(preds[1]))  #preds[1] = accuracy
preds = model.evaluate_generator(validation_generator, steps=175)
print ("validation Loss = " + str(preds[0]))
print ("validation Accuracy = " + str(preds[1]))
preds = model.evaluate_generator(test_generator, steps=175)
print ("test Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
#model=load_model(model_files[i])
    #print(model_files[i])   
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()          
###

print("DONE")

import os
import sys                                                                                                                                                                   
import time                                                                                                                                                                  
import numpy as np
os.environ["THEANO_FLAGS"]  = "mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=1"                                                                                                                                                           

import keras
from models_WGAN import *                                                                                                                                                 
from keras.utils import generic_utils                                                                                                                                        
sys.path.append("../utils")                                                                                                                                                  
import general_utils                                                                                                                                                         
import data_utils                                                                                                                                                            
import matplotlib.pyplot as plt                                                                                                                                          
from IPython import display  
import keras.backend as K                                                                                                                                                    
from keras.models import Model                                                                                                                                           
from keras.layers import Input,merge                                                                                                                                         
from keras import initializations                                                                                                                                            
from keras.utils import visualize_util                                                                                                                                       
from keras.layers.advanced_activations import LeakyReLU                                                                                                                      
from keras.activations import linear                                                                                                                                         
from keras.layers.normalization import BatchNormalization                                                                                                                    
from keras.layers.core import Flatten, Dense, Activation, Reshape, Lambda, Dropout                                                                                           
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D, MaxPooling2D                                                                            
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D                                                                                                    
from keras.layers.noise import GaussianNoise                                                                                                                                 
from keras.regularizers import *                                                                                                                                             
import resnet50
from keras.applications.vgg16 import VGG16

def vgg16(img_dim,n_classes,pretrained,wd, model_name="resnet"):
    drop=0.5
    input = Input(shape=(3,64,64), name="image_input")
    vgg16 = VGG16(include_top=False, weights='imagenet')
    x = vgg16(input)
    x = Flatten()(x)
    x = Dropout(0.5)(x) 
    out = Dense(n_classes, activation='softmax',init="he_normal", name='fc',W_regularizer=l2(wd))(x)
    vgg16_model = Model(input=input, output=out, name=model_name)                                                                                                           
    #model_path = "../../models/DCGAN"                                                                                                                                        
    #path = os.path.join(model_path, 'vgg16_OfficeDslrToAmazon.h5')                                                                                                           
    #vgg16_model.load_weights(path)                                                                                                                                           

    return vgg16_model

img_dim=64
image_dim_ordering='th'

X_source_train,Y_source_train,X_source_test, Y_source_test,n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='OfficeDslr')
X_dest_train,Y_dest_train,X_dest_test, Y_dest_test, n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='OfficeAmazon')

n_classes = n_classes1 #                                                                                                                                            
                                                                                                                                                              
img_source_dim = X_source_train.shape[-3:] # is it backend agnostic?                                                                  
img_dest_dim = X_dest_train.shape[-3:]

opt_C = data_utils.get_optimizer('SGD', 0.01)
classifier = vgg16(img_dest_dim,n_classes,pretrained=False,wd=0.0001)
classifier.summary()
model_path = "../../models/DCGAN"                                                                                               
class_weights_path = os.path.join(model_path, 'vgg16r_OfficeDslrToAmazon.h5')  
trained=False
if trained:
    classifier.load_weights(class_weights_path)
    
classifier.compile(loss='categorical_crossentropy', optimizer=opt_C,metrics=['accuracy'])

loss1,acc1 =classifier.evaluate(X_dest_test, Y_dest_test,batch_size=512, verbose=0)        
print('\n Classifier Accuracy on target domain test set before training: %.2f%%' % (100 * acc1))                                                              

sgd = keras.optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
K.set_value(sgd.lr,  K.get_value(sgd.lr))

if not trained:
    classifier.fit(X_dest_train, Y_dest_train, validation_split=0.01, batch_size=256, nb_epoch=30, verbose=1)                                                        
    K.set_value(sgd.lr, 0.1 * K.get_value(sgd.lr))
    classifier.fit(X_dest_train, Y_dest_train, validation_split=0.01, batch_size=256, nb_epoch=10, verbose=1)                                                        
loss2,acc2 = classifier.evaluate(X_dest_test, Y_dest_test,batch_size=512, verbose=0)
print('\n Classifier Accuracy on target domain test set after training:  %.2f%%' % (100 * acc2))
loss3, acc3 = classifier.evaluate(X_source_test, Y_source_test,batch_size=512, verbose=0)
print('\n Classifier Accuracy on source domain test set:  %.2f%%' % (100 * acc3))

model_path = "../../models/DCGAN"                                                                                               
class_weights_path = os.path.join(model_path, 'vgg16r_OfficeDslrToAmazon.h5')
classifier.save_weights(class_weights_path, overwrite=True)

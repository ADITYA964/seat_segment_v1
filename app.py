import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Dropout, Lambda,Conv2D, Conv2DTranspose,MaxPooling2D,concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from PIL import Image
from IPython.display import display
import streamlit as st
from tempfile import NamedTemporaryFile

import gdown
url="model_path")

output='model7bestttt.h5'
gdown.download(url, output, quiet=False)

import gdown
url="model_path")

output='model2.h5'
gdown.download(url, output, quiet=False)

IMG_WIDTH = 992
IMG_HEIGHT = 576 
IMG_CHANNELS = 3

import keras.backend as K
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)


c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (s)
#c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2),strides=2) (c1)

c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
#c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2),strides=2) (c2)

c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
#c3 = Dropout(0.1) (c3)
c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2),strides=2) (c3)

c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
#c4 = Dropout(0.1) (c4)
c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2),strides=2) (c4)

c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c4)

c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (p4)
#c5 = Dropout(0.1) (c5)
c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (u6)
#c6 = Dropout(0.1) (c6)
c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', padding='same') (u7)
#c7 = Dropout(0.1) (c7)
c7 = Conv2D(64, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', padding='same') (u8)
#c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (u9)
#c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time


import streamlit as st
from tempfile import NamedTemporaryFile

st.write("""
         # Count Seats using AI
         ## by Aditya Bagwadkar
         """
         )

file = st.file_uploader("Please upload snapshot of auditorium", type=["png","jpg"])
temp_file = NamedTemporaryFile(delete=False)         


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil 
from PIL import Image

if file is None:
  st.markdown("<div><span class='highlight green'><font color='white'>Please upload an image file in PNG or JPG format</font></span><div>", unsafe_allow_html=True)  
else:
  temp_file.write(file.getvalue())
  
  if not os.path.isdir("test"):
    os.mkdir("test")
    
  im = Image.open(temp_file)
  im.save("test/1.png")
  
  test_PATH=glob.glob('test/')
  X_ids = os.listdir(test_PATH[0])
  test = np.zeros((len(X_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
  sys.stdout.flush()
  x_ids=enumerate(X_ids)

  for n, id_ in x_ids:
      path = test_PATH[0] + id_
      img = imread(path)[:,:,:IMG_CHANNELS]
      img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      test[n] = img
  
  sample =test[0]

  model1 = Model(inputs=[inputs], outputs=[outputs])
  model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou_coef,'accuracy',recall,precision])
  model1.load_weights("model7bestttt.h5")

  img_pred1 = np.expand_dims(sample, axis=0)

  with tf.device('/GPU:0'):
      pred1=model1.predict(img_pred1, verbose=1)
  img1=(cv2.cvtColor(np.squeeze((np.round(pred1))*255), cv2.COLOR_GRAY2BGR)).astype(np.uint8) 

  model2 = Model(inputs=[inputs], outputs=[outputs])
  model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou_coef,'accuracy',recall,precision])
  model2.load_weights("model2.h5")

  img_pred2 = np.expand_dims(sample, axis=0)

  with tf.device('/GPU:0'):
      pred2=model2.predict(img_pred2, verbose=1)
  img2=(cv2.cvtColor(np.squeeze((np.round(pred2))*255), cv2.COLOR_GRAY2BGR)).astype(np.uint8) 
 
  img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
  contours_1, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  img1=cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
  img1 = cv2.drawContours(img1, contours_1, -1, (255,0,0), 3)

  img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
  contours_2, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
  img2 = cv2.drawContours(img2, contours_2, -1,(0, 183, 235), 3)

  RES = img1 + img2
  
  A=plt.figure(figsize=(30,20))
    
  plt.subplot(1,2,1)
  plt.axis("off")
  plt.imshow(Image.fromarray(sample))
  plt.grid(False)
    
  plt.subplot(1,2,2)
  plt.axis("off")
  plt.imshow(Image.fromarray(RES))
  plt.grid(False)

  st.pyplot(A)

  st.write("""## Occupied seats = """,len(contours_2))   
  st.write("""## Not Occupied seats = """,len(contours_1))   
  
  shutil.rmtree("test")  



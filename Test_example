import os
# os.chdir("C:/Users/")
import sys
# sys.path.insert(0,"C:/Users/")
import copy
import numpy as np
np.random.seed(121)
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

import keras
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import convolutional, pooling
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Reshape, merge, concatenate, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import average 
from keras.models import Input, Model
from keras import applications
from keras import optimizers, metrics
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model


## Importing images

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO




## Loading Data

data_imagetype = pd.read_csv("Sept_2017_Nov_2017.csv")

# # For Driver Front
# data_FP = data_imagetype[data_imagetype['IQAUXSEQ'] == 1]
# # data_manualratings = pd.read_csv("Final Ratings.csv")
# # For Passenger Front
data_RP = data_imagetype[data_imagetype['IQAUXSEQ'] == 1]
# data_RP = data_RP.sort_values(by='STAUCDAT')
# data_RD = data_RD[(data_RD['invoiceDate'] >= 20170101) & (data_RD['invoiceDate'] <= 20170630)]
# # data_FD = data_FD[0:40000]
data_RP = data_RP.reset_index(drop=True)
results = pd.DataFrame()

X_DmgData = data_RP['STDMGTYP']
X_DmgData[X_DmgData != "FR"] = 0
X_DmgData[X_DmgData == "FR"] = 1




## Load the corresponding model

ModelY = load_model('newmodelY_FP.h5')


## Extract image and convert to numpy array

%%time

## Preparing the X_data for Keras input
j = 0

def extracturl(x):
#     img = np.array(Image.open(BytesIO(requests.get(x).content)))
    ## Resize to 256*256
    global j
    print(j)
    j += 1
    try:
        img = img_to_array(load_img(BytesIO(requests.get(x).content), target_size=(256, 256)))
        return img
    except:
        img = img_to_array(load_img(BytesIO(requests.get('https://cs.copart.com/v1/AUTH_svc.pdoc00001/PIX85/fa1b4fe8-19ba-4345-b684-e826fd19cb11.JPG').content), target_size=(256, 256)))
        return img


## Rating the images

%%time

for i in range(6):
    temp_data = data_RP[i*7167:(i+1)*7167]
    print(i)
    X_data=[]
    X_data.append(temp_data["URL"].apply(extracturl))
    X_data = np.array(X_data, dtype=np.uint8)
    X_data = np.squeeze(X_data, axis=(0,))
    X_temp_DmgData = X_DmgData[i*7167:(i+1)*7167]
    Y_predict = ModelY.predict([X_data, X_temp_DmgData])
    temp_data["105 Passenger Front"] = Y_predict[:,0]
    temp_data["110 R"] = Y_predict[:,1]
    temp_data["114 R"] = Y_predict[:,2]
    temp_data["117 Passenger Front"] = Y_predict[:,3]
    temp_data["120 R"] = Y_predict[:,4]
    results = results.append(temp_data, ignore_index=True)
    
   results.to_csv('UK_Sept17_Nov17_FP.csv')

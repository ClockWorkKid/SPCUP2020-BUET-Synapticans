####### THIS IS CODE TO TRAIN ANOMALOUS IMAGE IN ISOLATION FOREST APPROACH #########
#first some dependencies needed to be installed like 
#pip install Keras
#pip install pandas
#pip install opencv-contrib-python
#pip install matplotlib
#pip install pickle-mixin
#pip install scikit-learn
#pip install numpy

from sklearn.ensemble import IsolationForest
import argparse
import pickle
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",type=str,default="collective normal/",    #name the all normal(training) image folder
	help="path to dataset of images")
ap.add_argument("-m", "--model",type=str,default="trained_model.model",    #In which name the trained model will be saved
	help="path to output anomaly detection model")
args = vars(ap.parse_args())

# load and quantify our image dataset
print("[INFO] preparing dataset...")

##PREPARING FEATURE###
from keras.layers import *
from keras.preprocessing import image
import numpy as np

import  numpy  as  np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from keras.applications import DenseNet201

#to train on other feature extractor
'''
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
#from keras.applications.resnet152 import ResNet
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
'''
def frame_from_dir(dir):
    temp=os.listdir(dir)   #give input directory
    video=[]
    for i in temp:
        if(i.endswith('.JPG')):
            video.append(i)
    return video


cur_dir=os.getcwd()
input_path=os.path.join(cur_dir,args["dataset"])         #whatever test_directory is named
frame=frame_from_dir(input_path)
data=[]

model = DenseNet201(weights='imagenet', include_top=False)
for xy in range(len(frame)):   #frame by frame passed through feature extractor to extract feature
    img_path = os.path.join(input_path,frame[xy])
    img = image.load_img(img_path, target_size=(530, 700))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)    
    vgg_feature = model.predict(img_data)
    data.append(vgg_feature)

a=np.array(data)
a=a.reshape(len(frame),-1)   #converting to a feature vector

# train the anomaly detection model
print("[INFO] fitting anomaly detection model...")

modelanom = IsolationForest(n_estimators=150, contamination=0.01,
	random_state=42)

modelanom.fit(a)

# serialize the anomaly detection model to disk
f = open(args["model"], "wb")
f.write(pickle.dumps(modelanom))
f.close()

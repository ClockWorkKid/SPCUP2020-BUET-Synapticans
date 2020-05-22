####### THIS IS CODE TO TEST ANOMALOUS IMAGE IN ISOLATION FOREST APPROACH #########
#first some dependencies needed to be installed like 
#pip install Keras
#pip install pandas
#pip install opencv-contrib-python
#pip install matplotlib
#pip install pickle-mixin
#If you don't train a model then you can download trained model weight from : https://drive.google.com/open?id=1pKVodmGifGudE758X9-XG8v376DPFWKu
#then place 'anomaly_detectornedense201.model' in this directory to use it

import argparse
import pickle
import cv2
import os
import pandas as pd
import time
starting=time.time()

# construct the argument parser and parse the arguments
cur_dir=os.getcwd()
modelpath=os.path.join(cur_dir,"anomaly_detectornedense201.model") 
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",type=str,default=modelpath,
	help="path to trained anomaly detection model")
ap.add_argument("-p", "--path",type=str,default="imagesA2",     #folder name of anomalous frames,all testing frame must be in a folder 
	help="path to input image")
ap.add_argument("-im", "--show",type=int,default=0,     #0 to not show frame by frame,1 to show frame by frame like video 
	help="path to input image")
args = vars(ap.parse_args())

print("[INFO] loading Feature Extractor...")
from keras.applications import DenseNet201      #first run Densenet201 pretrained weight will download,it will take some times
model = DenseNet201(weights='imagenet', include_top=False)
# load the anomaly detection model
print("[INFO] loading anomaly detection model...")    #loading isolation forest model
modelanom = pickle.loads(open(args["model"], "rb").read())


'''Feature extraction'''

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input


def frame_from_dir(dir):
    temp=os.listdir(dir)   #give input directory
    video=[]
    for i in temp:
        if(i.endswith('.JPG')):
            video.append(i)
    return video

cur_dir=os.getcwd()
input_path=os.path.join(cur_dir,args["path"])         #whatever test_directory is named
frame=frame_from_dir(input_path)
data=[]
for xy in range(len(frame)):
    img_path = os.path.join(input_path,frame[xy])
    img = image.load_img(img_path, target_size=(530, 700))
    #img = image.load_img(img_path, target_size=(331, 331))  #for nasnetlarge
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = model.predict(img_data)
    print(feature.shape)
    data.append(feature)
a=np.array(data)
a=a.reshape(len(frame),-1)


anom=[]
prd=[]
for xy in range(len(frame)):
    image = cv2.imread(os.path.join(input_path,frame[xy]))
    width = 750
    height = 450
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    features = a[xy]
    preds = modelanom.predict([features])[0]     #for isolation forest
    prd.append(preds)
    scores_pred = (modelanom.decision_function(features.reshape(1,-1))*-1)+0.5   #in isolation forest at first setup th is about 0.066.Smaller than this will be anomaly
    anom.append(scores_pred)
    
    
anom=np.array(anom)
if np.mean(anom)>0.4:   #as threshold data seems to biased at mean of 0.4 when more abnormal is given than normal
    anom=(anom - np.min(anom))/np.ptp(anom)
else:
    anom=abs(1.4*anom**3-2.3*anom**2)  #empirical sqashing function


#If you want to see franewise score  
for xy in range(len(frame)):
    image = cv2.imread(os.path.join(input_path,frame[xy]))
    width = 750
    height = 450
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    label = "anomaly" if anom[xy]>0.5 else "normal"
    color = (0, 0, 255) if anom[xy]>0.5 else (0, 255, 0)
    # draw the predicted label text on the original image
    cv2.putText(image, label+str(anom[xy]), (10,  25), cv2.FONT_HERSHEY_SIMPLEX,
    	0.8, color, 2)
    # display the image if you want ot see frmae annotated with score
    if args["show"]:
        cv2.imshow("Output", image)
        cv2.waitKey(400)
        cv2.destroyAllWindows()

ending=time.time()
print(f"Execution time is {ending-starting}")
import pandas as pd 
pd.DataFrame(anom).to_csv("isofor_image_score.csv",header=None, index=None)
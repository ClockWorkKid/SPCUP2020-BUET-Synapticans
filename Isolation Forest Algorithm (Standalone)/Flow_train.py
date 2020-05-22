from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor
import argparse
import pickle
import os
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",type=str,default="collective normal/",
	help="path to dataset of images")
ap.add_argument("-m", "--model",type=str,default="anomaly_detectorincresV2OF.model",
	help="path to output anomaly detection model")
args = vars(ap.parse_args())

##PREPARING FEATURE###
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import  numpy  as  np
from keras.applications.resnet50 import preprocess_input
#from keras.applications import DenseNet121
from keras.applications import DenseNet201
#from keras.applications.nasnet import NASNetLarge
#from keras.applications.xception import Xception
#from keras.applications.resnet import ResNet101
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.inception_v3 import InceptionV3
def frame_from_dir(dir):
    temp=os.listdir(dir)   #give input directory
    video=[]
    for i in temp:
        if(i.endswith('.JPG')):
            video.append(i)
    return video

WIDTH = 700
STEP = 16
QUIVER = (255, 100, 0)

def draw_flow(img, flow, step=STEP):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, QUIVER)
    
    return vis


cur_dir=os.getcwd()
input_path=os.path.join(cur_dir,args["dataset"])         #whatever test_directory is named
frame=frame_from_dir(input_path)
data=[]

#model=InceptionV3(include_top=False, weights='imagenet')
#model=Xception(include_top=False, weights='imagenet')
#model=NASNetLarge(include_top=False, weights='imagenet')
#model=InceptionResNetV2(include_top=False, weights='imagenet')
model = DenseNet201(weights='imagenet', include_top=False)
#model=ResNet101V2(weights='imagenet', include_top=False)

img_path = os.path.join(input_path,frame[0])
#img_path = 'G:/SP CUP 2020/Anomaly From Image/intro-anomaly-detection/collective normal/1574069785973.JPG'
img=cv2.imread(img_path)
prev = cv2.resize(img, (700,530), interpolation = cv2.INTER_AREA)
prevgray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
#vis = cv2.cvtColor(prevgray, cv2.COLOR_GRAY2BGR)
img_data=prev.astype("float32")
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)    
vgg_feature = model.predict(img_data)
data.append(vgg_feature)

for xy in range(1,len(frame)):
    img_path = os.path.join(input_path,frame[xy])
    #img_path = 'G:/SP CUP 2020/Anomaly From Image/intro-anomaly-detection/collective normal/1574069785973.JPG'
    img=cv2.imread(img_path)
    img = cv2.resize(img, (700,530), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    flowimg=draw_flow(gray, flow)
    #cv2.imshow('flow',flowimg)
    #k=cv2.waitKey(0)
    #if k=='q':
    #    break
    #cv2.destroyAllWindows()
    #cv2.imwrite(f"/G:SP CUP 2020/Anomaly From Image/intro-anomaly-detection/flow/{xy}.jpg",flowimg)
    #img_path = os.path.join(input_path,frame[xy])
    #img_path = 'G:/SP CUP 2020/Anomaly From Image/intro-anomaly-detection/collective normal/1574069785973.JPG'
    #img = image.load_img(img_path, target_size=(530, 700))
    #img = image.load_img(img_path, target_size=(331, 331))  #for nasnetlarge
    #img_data = image.img_to_array(img)
    img_data=flowimg.astype("float32")
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)    
    vgg_feature = model.predict(img_data)
    data.append(vgg_feature)

a=np.array(data)
a=a.reshape(len(frame),-1)
# train the anomaly detection model
print("[INFO] fitting anomaly detection model...")

modelanom = IsolationForest(n_estimators=150, contamination=0.01,
	random_state=42)
'''
modelanom = LocalOutlierFactor(n_neighbors=35, contamination=0.01,novelty=True)   #for LOF approach
'''
modelanom.fit(a)

# serialize the anomaly detection model to disk
f = open(args["model"], "wb")
f.write(pickle.dumps(modelanom))
f.close()
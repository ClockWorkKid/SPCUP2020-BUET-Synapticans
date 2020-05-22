import argparse
import pickle
import cv2
import os
import pandas as pd
import time
starting=time.time()
#from sklearn.neighbors import LocalOutlierFactor    #for LOF
os.chdir("G:/SP CUP 2020/Anomaly From Image/intro-anomaly-detection")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",type=str,default="anomaly_detectordense201AngImg700.model",       #name of pretrained model
	help="path to trained anomaly detection model")
ap.add_argument("-i", "--image",type=str,default="collective anomaly/1579257477699.jpg", 
	help="path to input image")
ap.add_argument("-p", "--path",type=str,default="imagesB3",     #test image folder
	help="path to input image")
args = vars(ap.parse_args())


# load the anomaly detection model
print("Now loading anomaly detection model...")
modelanom = pickle.loads(open(args["model"], "rb").read())


'''Feature extraction'''
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from keras.applications import DenseNet201
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.nasnet import NASNetLarge
#from keras.applications.resnet_v2 import ResNet50V2


#model = InceptionResNetV2(weights='imagenet', include_top=False)
model = DenseNet201(weights='imagenet', include_top=False)
#model=InceptionV3(include_top=False, weights='imagenet')   
#model=NASNetLarge(include_top=False, weights='imagenet')
#model = VGG16(weights='imagenet', include_top=False)
#model=ResNet50V2(include_top=False, weights='imagenet')
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

img_path = os.path.join(input_path,frame[0])
#img_path = 'G:/SP CUP 2020/Anomaly From Image/intro-anomaly-detection/collective normal/1574069785973.JPG'

frame1 = image.load_img(img_path, target_size=(700, 700))    ##try with 224*224(ideal size of densnet)
frame1 = image.img_to_array(frame1)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

for xy in range(len(frame)):
    img_path = os.path.join(input_path,frame[xy])
    #img_path = 'G:/SP CUP 2020/Anomaly From Image/intro-anomaly-detection/collective normal/1574069785973.JPG'
    img = image.load_img(img_path, target_size=(700, 700))
    img = image.img_to_array(img)
    #img = image.load_img(img_path, target_size=(331, 331))  #for nasnetlarge
    next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    #flow = optical_flow.calc(prvs, next, None)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.8, 3, 10, 3, 7, 1.1, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    prvs = next
    
    #img_data = image.img_to_array(img)
    img2 = np.zeros_like(img)
    img2[:,:,0] = ang
    img2[:,:,1] = ang
    img2[:,:,2] = ang
    ##addition
    feed=img2+img    
    img_data = np.expand_dims(feed, axis=0)
    img_data = preprocess_input(img_data)    
    vgg_feature = model.predict(img_data)
    data.append(vgg_feature)
a=np.array(data)
a=a.reshape(len(frame),-1)

anom=[]
prd=[]
for xy in range(len(frame)):
    image = cv2.imread(os.path.join(input_path,frame[xy]))
    width = 400
    height = 300
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    features = a[xy]
    #preds = modelanom.fit([features]).predict([features])  #was for outlier detection which is not needed
    #preds = modelanom.predict([features])  #for LOF,for novelty detection
    preds = modelanom.predict([features])[0]     #for isolation forest
    prd.append(preds)
    scores_pred = modelanom.decision_function(features.reshape(1,-1))*-1   #in isolation forest at first setup th is about 0.066.Smaller than this will be anomaly
    #scores_pred=(scores_pred - np.min(scores_pred))/np.ptp(scores_pred)
    anom.append(scores_pred)
    label = "anomaly" if preds == -1 else "normal"
    color = (0, 0, 255) if preds == -1 else (0, 255, 0)
    # draw the predicted label text on the original image
    cv2.putText(image, label+str(scores_pred+.2), (10,  25), cv2.FONT_HERSHEY_SIMPLEX,
    	0.7, color, 2)
    # display the image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#plotting anomaly score to localize in time domain
    
anom=np.array(anom)
if np.mean(anom)>0.4:   #as threshold data seems to biased at mean of 0.4 when more abnormal is given than normal
    anom=(anom - np.min(anom))/np.ptp(anom)
else:
    anom=abs(1.4*anom**3-2.3*anom**2)  #empirical sqashing function


plt.plot(anom) # plotting by columns
plt.xticks(np.arange(0,len(anom),3))
plt.ylabel("Anomaly score")
plt.show()
ending=time.time()
print(f"Elapsed time is {ending-starting}")
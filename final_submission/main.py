from devnet_utility import *
from isoforest_func import isolation_forest_score
import pandas as pd
from keract import get_activations
import os
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings("ignore")

#devnet architecture running function takes a dataframe of IMU data as parameter and
# returns a 1D array of anomaly score which contains all the anomaly score according to the timestamp.
def devnet(df, csv_name):
	df = df.drop(["time"], axis=1)
	x=df.values # getting input value
	input_shape=x.shape[1:] # getting input shape
	model=deviation_network(input_shape) #defining the model
	model.load_weights("./model/devnet_final.h5") #loading weight file



	most_responsible_sensor_dict ={
	    "ori_x":0,"ori_y":0,"ori_z":0,"ori_w":0,"vel_ang_x":0,"vel_ang_y":0,"vel_ang_z":0,"acc_x":0,"acc_y":0,"acc_z":0
	}
	all_sensor_name = ["ori_x","ori_y","ori_z","ori_w","vel_ang_x","vel_ang_y","vel_ang_z","acc_x","acc_y","acc_z"]
	annomaly_counter=0
	final_pred = np.array([])
	for each_instance in x:
	    each_instance= each_instance.reshape(1,10)
	    score = model.predict(each_instance)
	    final_score = (1/(0.625+np.exp(-abs(1.3*score))))-(1/1.625)
	    final_pred = np.append(final_pred, final_score[0][0])
	    
	    if final_score >= 0.5:
	        annomaly_counter+=1
	        activations = get_activations(model, each_instance, layer_name="hl1")
	        hidden_layer_value = activations[list(activations.keys())[0]]
	    
	    
	        w1 = K.get_value(model.get_layer('hl1').weights[0])
	        w2 = K.get_value(model.get_layer('score').weights[0])
	    
	        weighted_hidden_layer=w2.reshape(20)*hidden_layer_value.reshape(20)
	    
	    
	        high_hidden_node_list = np.argsort(abs(weighted_hidden_layer))[-5:]
	        high_hidden_node_list=np.flip(high_hidden_node_list)
	        for i in high_hidden_node_list:
	            if weighted_hidden_layer[i] == 0:
	                idx = np.where(high_hidden_node_list==i)
	                high_hidden_node_list = np.delete(high_hidden_node_list, idx)
	            
	    
	        sensor_list=np.array([])
	        for i in high_hidden_node_list:
	            sensor_list = np.append(sensor_list, np.argsort(w1[:,i])[-3:]) # getting the most activated sensors list for high anomaly score
	    
	    
	    
	        for j,i in enumerate(sensor_list):
	            most_responsible_sensor_dict[all_sensor_name[int(i)]] += len(sensor_list)-j #giving score to sensor in order to rank
	    
	    
	    




	

	most_responsible_sensor_dict = {k: v for k, v in sorted(most_responsible_sensor_dict.items(), key=lambda item: item[1], reverse=True)} 

	responsible_sensor=[]
	for i,j in enumerate(most_responsible_sensor_dict.keys()):
		if i==0 and most_responsible_sensor_dict[j] == 0:
			break
		elif i==3:
		    break
		else:
			responsible_sensor.append(j)

	if len(responsible_sensor) == 3:
		os.makedirs('reports', exist_ok=True)
		with open(f'reports/{csv_name}.txt', 'w') as f:
			print(f"The most responsible sensor for anomaly instances in {csv_name} are (from highest to lowest) : ", file=f)
			for sensor in responsible_sensor:
				print(sensor, file=f)
	    



	final_pred = final_pred.reshape(len(final_pred))

	return final_pred
	    







#Ensemble of devnet and isolation forest, weights are computed by linear regression algorithm
def ensemble(devnet_score, iforest_score,time, csv_name):

	final_score = (0.6861 * iforest_score) + (0.537 * devnet_score)
	final_score = np.clip(final_score, 0, 1)

	final_score = final_score.reshape(len(final_score))

	final_df = pd.DataFrame()
	final_df["time"] = time
	final_df["score"] = final_score


	os.makedirs("results", exist_ok=True)
	final_df.to_csv(csv_name, index=False)

	return final_score






if __name__ == "__main__":

	csv_names = []

	for filename in listdir("csv"):
		if filename.endswith(".csv"):
			csv_names = csv_names + [filename[0:len(filename)-4]]


	for filename in csv_names:

		df = pd.read_csv('csv/'+filename+'.csv')
		timevals = df.time
		dev_score = devnet(df, filename)
		isofr_score=isolation_forest_score(df) #anomaly score vector
		final_score = ensemble(dev_score,isofr_score,timevals,'results/'+filename+'.csv')

	
	print("\n\n\n\n\n")
	print("Necessary results and reports are saved in './results/' and './reports/' directory accodingly")
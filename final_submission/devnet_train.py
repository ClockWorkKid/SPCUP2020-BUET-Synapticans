# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.sparse import vstack, csc_matrix
from utils_train_dev import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import time

MAX_INT = np.iinfo(np.int32).max
data_format = 0

def dev_network_d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
    return Model(x_input, intermediate)

def dev_network_s(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
    return Model(x_input, intermediate)

def dev_network_linear(input_shape):
    '''
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    '''    
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
    return Model(x_input, intermediate)

def deviation_loss(y_true, y_pred):
    '''
    z-score-based deviation loss
    '''    
    confidence_margin = 5.     
    ## size=5000 is the setting of l in algdev_network_linearorithm 1 in the paper
    ref = K.variable(np.random.normal(loc = 0., scale= 1.0, size = 5000) , dtype='float32')
    dev = (y_pred - K.mean(ref)) / K.std(ref)
    inlier_loss = K.abs(dev) 
    outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
    return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


def deviation_network(input_shape, network_depth):
    '''
    construct the deviation network-based detection model
    '''
    if network_depth == 4:
        model = dev_network_d(input_shape)
    elif network_depth == 2:
        model = dev_network_s(input_shape)
    elif network_depth == 1:
        model = dev_network_linear(input_shape)
    else:
        sys.exit("The network depth is not set properly")
    rms = RMSprop(clipnorm=1.) # RMSprop is the optimizer
    model.compile(loss=deviation_loss, optimizer=rms)
    return model


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:                
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
            # taking random "normal" samples from the x_train(i.e x) and their corresponding training label
            # both the variables are returned as numpy array
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield(ref, training_labels) # yield makes the function as a generator and a generator can be iterated through
        # yield makes the function returns something without finishing the function or completing the function
        # generator e infite loop thakle (mane infinite amount data generate kortese) jokhon code sesh hobe tokhon loop sesh hobe
        # so CPU tee batch generate hote thake and keras GPU tee seita niye training koray
        if (counter > nb_batch):
            counter = 0
 
def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
	        sid = rng.choice(n_inliers, 1)
	        ref[i] = x_train[inlier_indices[sid]]
	        training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

 
def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    '''      
    ref = np.empty((batch_size))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.array(training_labels)


def load_model_weight_predict(model_name, input_shape, network_depth, x_test, y_test):
    '''
    load the saved weights to make predictions
    '''
    model = deviation_network(input_shape, network_depth) # defining the network
    model.load_weights(model_name) # loading the weight of the network
    print(model.summary())
    scoring_network = Model(inputs=model.input, outputs=model.output) # apatoto toh mone hocche naah kono bisesh kaaj kortese eita 
    
    if data_format == 0:
        scores = scoring_network.predict(x_test) # generating scores
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


def inject_noise_sparse(seed, n_out, random_seed):  
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    '''
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    print(noise.shape)
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()

def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''  
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise

def run_devnet(args):
    names = args.data_set.split(',') # taking the dataset name
    #names = ['annthyroid_21feat_normalised']
    network_depth = int(args.network_depth) #selecting the network depth
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs #how many times we repeat the experiments to obtain the average performance
        rauc = np.zeros(runs) # defining evaluation metric
        ap = np.zeros(runs)
        filename = nm.strip() #getting the dataset filename
        global data_format # initially it is set to zero
        data_format = int(args.data_format) # 0 for csv and 1 for lbsvm
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()    
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices] # getting the outliers input
        n_outliers_org = outliers.shape[0] # number of outlier in the dataset
        
        train_time = 0
        test_time = 0
        for i in np.arange(runs):  
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.01, random_state=42, stratify = labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0] #getting the outlier/annomaly data indexes
            inlier_indices = np.where(y_train == 0)[0] ##getting the normal data indexes
            x_test = np.append(x_test, np.expand_dims(x_train[outlier_indices[0]], axis=0), axis=0)
            y_test = np.append(y_test, np.expand_dims(y_train[outlier_indices[0]], axis=0), axis=0) # so that the test data has at least one annomalous data

            n_outliers = len(outlier_indices) # number of annomaly data
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            
            n_noise  = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise) # amount of noise data or contaminated in the normal data
            

            rng = np.random.RandomState(random_seed)  
            if data_format == 0:                
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers # number of unknown annomalies
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)            
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)
                
                noises = inject_noise(outliers, n_noise, random_seed) #creating noises
                x_train = np.append(x_train, noises, axis = 0) # adding the noise in the training data
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1))) # labeling the noise as normal data

                #number of unknwn annomalies are replaced in the training data as noise and their class label is set to normal data class label
            
            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)        
                    retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                    retain_idx = list(retain_idx)
                    x_train = x_train[retain_idx]
                    y_train = y_train[retain_idx]                               
                
                noises = inject_noise_sparse(outliers, n_noise, random_seed)
                x_train = vstack([x_train, noises]) # row-wise stacking the metrices
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            outlier_indices = np.where(y_train == 1)[0] # now this gives the known annomalies indexes
            inlier_indices = np.where(y_train == 0)[0] # normal data indexes after introducing noises
            print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
            input_shape = x_train.shape[1:] # dimension of the input
            n_samples_trn = x_train.shape[0] # total input/training data
            n_outliers = len(outlier_indices) # number of annomaly data in the input
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            
            
            
            start_time = time.time() 
            input_shape = x_train.shape[1:] # dimension of the input verctor
            epochs = args.epochs #the number of epochs
            batch_size = args.batch_size   #batch size used in SGD 
            nb_batch = args.nb_batch  #the number of batches per epoch
            model = deviation_network(input_shape, network_depth)
            print(model.summary())  
            model_name = "./model/devnet_"  + filename + "_spcup" + str(args.batch_size) +"bs_" + str(args.known_outliers) + "ko_" + str(network_depth) +"d_"+'round_' + str(i)+".h5"
            
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=1,
                                           save_best_only = True, save_weights_only = True) #in order to save the best model     
            
            model.fit_generator(batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                                          steps_per_epoch = nb_batch,
                                          epochs = epochs,
                                          callbacks=[checkpointer]) # training is started  
            train_time += time.time() - start_time
            
            start_time = time.time()

            scores = load_model_weight_predict(model_name, input_shape, network_depth, x_test, y_test) # generating the scores
            

            print("\n\n\n")
            print("The testing score are \n")
            print(scores)
            print("\n\n\n")
            test_time += time.time() - start_time
            rauc[i], ap[i] = aucPerformance(scores, y_test) # generating the evaluation metrices    
        
            #time.sleep(25)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time/runs
        test_time = test_time/runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))    
        print("average runtime: %.4f seconds" % (train_time + test_time))
        writeResults(filename+'_'+str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org, n_outliers,
                      network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)
        


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--network_depth", choices=['1','2', '4'], default='2', help="the depth of the network architecture")
	parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
	parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
	parser.add_argument("--epochs", type=int, default=100, help="the number of epochs")
	parser.add_argument("--runs", type=int, default=20, help="how many times we repeat the experiments to obtain the average performance")
	parser.add_argument("--known_outliers", type=int, default=17, help="the number of labeled outliers available at hand")
	parser.add_argument("--cont_rate", type=float, default=0, help="the outlier contamination rate in the training data") # this parameter was not used in our training although it was used specified in the DevNet paper
	parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
	parser.add_argument("--data_set", type=str, default='train', help="a list of data set names")
	parser.add_argument("--data_format", choices=['0','1'], default='0',  help="specify whether the input data is a csv (0) or libsvm (1) data format")
	parser.add_argument("--output", type=str, default='./results/result.csv', help="the output file path")
	parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
	args = parser.parse_args()
	run_devnet(args)


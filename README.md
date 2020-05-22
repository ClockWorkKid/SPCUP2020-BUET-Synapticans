# SPCUP2020-BUET-Synapticans
This repository contains the project files containing our approach towards solving the problem statement of IEEE Signal Processing Cup 2020

There are four main parts to the project, 'final_submission' contains all the final versions of the codes, an overall solution process to the project. 'Isolation forest (standalone)' contains all necessary files to implement isolation forest alone on a dataset to solve problem. In order to understand how to use the files, head to that folder and see the README

The 'Technical Report.pdf' file contains details of our algorithm for solving the problem statement and relevent experimentations, result and discussion.  


# IEEE Signal Processing Cup 2020
This year main problem was 'Unsupervised abnormality detection by using intelligent and heterogeneous autonomous systems'. More details is in the [link](https://signalprocessingsociety.org/get-involved/signal-processing-cup) and dataset is available [here](https://piazza.com/ieee_sps/spring2020/spcup2020/home). To detect anomalies and quantify each instance from a multimodal dataset we proposed an algorithm which is an Ensemble of Isolation Forest and DevNet.It was the 5th positioned solution in the competition.

A summary and overview of the solved problem is illustrated below.

![Result on IMU data](illustration/annomaly score for datasetB1csv.jpg)

Anomaly score generated from IMU data by both unsupervised algorithm, Isolation forest and DevNet is displayed on above figure. This was generated on one abnormal rosbag. 

![](Illustration/3wvqgp.gif)

Anomaly score generated from Image dataset of another abnormal rosbag.

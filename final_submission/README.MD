**Step 1:** 
	Install python version 3.6 from www.python.org 
	(During installation please select to install pip package from the custom option)
	Alternative: run the following section in a new virtual environment

**Step 2:**
	Open command prompt (for windows) or terminal (for Ubuntu) and enter the following command
	to install all necessary python packages required to run the network

	pip install keract keras tensorflow==1.14 pandas sklearn scikit-learn==0.21.2 pickle-mixin
	
**Step 3:**
	Paste the bag files to be analyzed inside 'bag' folder.
	Make sure the two folders 'csv' and 'results' are present.

**Step 4:**
	Run Matlab (R2019a or higher version) and from the editor menu, run the 'read_ros_bag.m' file
	Wait till command window shows "DONE"
	Csv files corresponding to each bag file should be generated in the 'csv' folder

**Step 5:**
	Go to the submitted folder and open command prompt there. 
	Write the following command and hit enter

	python main.py

**Step 6:**
	After Step 5 , anomaly scores per timestamp will be stored in a csv file corresponding to each
	bag file in the "results" directory and a general report will be printed in "reports" directory



# Reference 

1.Isolation Forest
https://ieeexplore.ieee.org/abstract/document/4781136/?casa_token=mql_bhbOVuUAAAAA:mQpuMPujitQT7vU05vLhBr0qJtEPifgNvuXjlhoY0v2KfIFfo40FKR3wjsRUVtXyVyMiauynx-OUog

2.DevNet
https://arxiv.org/pdf/1911.08623.pdf

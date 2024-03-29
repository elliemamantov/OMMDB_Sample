# OMMDB Sample Code
(***Incomplete!!***) Data processing pipeline and modeling code for the OMMDB project. \
Authors: Ellie Mamantov and Kayla Matheus

# Data Processing Pipeline
Processes radar, RGB Camera, respiration belt, motors, and IMU data

## bagpy_processing.py
Initial processing of raw bag files

**Input:** ROS bag files\
**Output:** Folder, titled with the bag name, containing csvs of each data stream in the current working directory of the bagfile(s)\
**Arguments:** \
		-*root*: root file path for the bag files\
		- *bagFiles*: bag file names\
		- *omit*: any bag files to remove from processing

## filter_respiration.py
Prepare the respiration belt data for labeling. After filtering, please label the data in label-studio and export labels in the JSON_MIN format

**Input:** respiration csvs, generated by bagpy_processing.py\
**Output:** New csv files for each breathing session with raw and filtered respiration data \
**Arguments:** \
		- *filepaths*: file paths of csvs you want to process\
		- *sigma*: the level of smoothing for the gaussian filtering\
		- *root*: root file path\
		- *alsoSave*: separate folder to save new csvs to, if wanted\
		- *bagRoot*: root path to bagfiles folder, to save the respiration data into session folders

## process_json.py
Check the human-created label data (check for wrong orders, large gaps between labels, etc.), then add new respiration labels to individual session folders  

**Input**: JSON file generated in label-studio\
**Output**: Will print alerts is any human errors need to be corrected, then adds resp_label.csv, a csv with the human-created labels, to the sessions folders\
**Arguments:**\
	- *filepaths*: file paths for the json files you want to process\
	- *gap*: the largest gap between human labels you will allow\
	- *annotator*: name of human labeler\
	- *saveRoot*: path to folder to save annotations to

## downsample_data.py
Downsamples processed data to a given Hz, 

**Input**: Session folders that contain the processed data created by scripts above\
**Output**: HDF5 files of each data type for each session: respiration_data_xhz.h5, radar_data_xhz.h5, images_data_xhz.h5, motion_data_xhz.h5\
**Arguments**: \
	- *folders*: names of the session folders holding the processed data\
	- *omit*: any folders you want to omit from this processing\
	- *root*: root file path for session folders\
	- *saveRoot*: file path for the folder to hold the output h5 files\
	- *hz*: hz to downsample to


## make_optical_flow.py
Preprocess the RGB camera input to create optical flow images using OpenCV's implementation of the Farneback algorithm

**Input**: Image files \
**Output**: Optical flow data for each session specified \
**Arguments**:\
	 - *folders*: session folders you want to process\
	 - *omit*: any folders you want to omit from this processing\
	 - *root*: root file path to the session folders are located\
	 - *hz*: downsample rate that corresponds to the folder you want to pull the data from\
	 - *h5*: whether you want to create h5 files of the optical flow data\
	 - *images*: whether you want to create a folder of the optical flow data

# Modeling 

## model_training.py

Script to train models according to many possible arguments, such as learning rate, number of convolutional features, dropout and regularization, and which local gpus to run on. Please see script for list of arguments. 

Also includes tensorboard logging and results tracking, including accuracy, F1 score, and confusion matrices

**Regions**:\
	- *Libraries*: Import all libraries necessary\
	- *Args parsing and variables setup*: parse command line arguments that detail what model to create and how it should be trained\
	- *Setup GPU*: Setup the use of local GPUs\
	- *Setup Datasets*: Use custom definitions from dataset.py to handle the datasets\
	- *Initialize Models*: Define the model according to the supplied motion states \
	- *Setup Data Splitting*: Create PyTorch Dataloaders for train, validation, and test sets\
	- *Training Function*: Defines what happens at each epoch of the training cycle, including evaluating the model at the end of each epoch\
	- *Evaluation Functions*: Define how the model will be evaluated at the end of each epoch\
	- *Training commands and Email*: Main call to training function, set up to email upon training completion

## dataset.py

Custom dataset definitions to handle the OMMDB custom dataset\
	- *VideoIndex_Stacked*: An indexing system to link an optical flow frame to an index. \
	- *OF_Dataset_sequenced_sliding*: The dataset class definition that utilizes a sliding window approach to extract datapoints from the data stream. 

## models.py

Definitions of all models tested for the project. \
	- *Full Model Both*: The entire model, including the chosen CNN, the IMU MLP, body motors input, and LSTM. Both refers to both types of motion input (IMU and Body motors) being included. This is the final model evaluated in the paper\
	- *Full Model No Motion*: The entire model, but with neither the IMU MLP or the body motors input. Therefore, it solely includes the chosen CNN and LSTM\
	- *Full Model Body Motion*: The entire model, but with only the body motors input. Therefore, it includes the chosen CNN, the body motors input, and the LSTM\
	- *Full Model IMU*: The entire model, but with only the IMU MLP. Therefore, it includes the chosen CNN, the IMU MLP, and the LSTM\
	- *CNN simple*: A simple CNN\
	- *CNN 2*: A simple CNN, but with Batch Normalization and lazy linear fully connected layers\
	- *IMU Net*: A simple MLP that increases the dimensionality of the raw IMU inputs

## normalize_optical_flow.py

Gets the mean and standard deviation of the two channels of the optical flow data, for the given participant IDs


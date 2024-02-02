import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
import deepdish as dd
from skimage.transform import resize
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


'''
Class that holds a dictionary of {number: "path/to/npz/file"} to allow for indexing the dataset
'''
class VideoIndex_Stacked():
    #pids is a list of participant ids to include 
    #does not include the first _stacksize_ frames of each session, since we cannot stack the ones before that
    def __init__(self, pids, seq_cap, stack_size, img_folder_name, sessions_obj, window_step = 3, sessions = None):
        self.to_pids_file_path = '/data/ommdb/10Hz'
        self.index = {}

        omits = ['']
        idx = 0
        first = True
        for session in os.listdir(self.to_pids_file_path):
            #Determine whether this session should be included in this index
            include = False
            if pids and sessions == None: #if using crossval by PID
                for pid in pids:
                    if pid in session:
                        include = sessions_obj.include_session(session)
            elif sessions: #if using 9fold crossval by sessions
                if session in sessions:
                    include = True

            if include:
                #add the session to the index
                directory_path = self.to_pids_file_path + "/" + session
                file_path = directory_path + "/" + img_folder_name + "/"
                samples = os.listdir(file_path)
                sample_path = samples[0]
                last_underscore = sample_path.rfind("_")
                last_period = sample_path.rfind('.')
                true_sample_number = sample_path[last_underscore+1:last_period]
                img_name_header = sample_path[:last_underscore+1]
                num_samples = len(samples)
                
                for i in range(stack_size, seq_cap, window_step):
                    if int(i) >= stack_size:
                        if i < seq_cap and i <= num_samples:
                            sample = file_path + img_name_header + str(i) + '.npz'
                            self.index[idx] = sample
                            idx = idx + 1



class OF_Dataset_sequenced_sliding(Dataset):
    # Make dataset of OF images sequences up to a sequence cap for the lSTM. Utilizes saved h5 files of the following outline: 
    '''
            flow_data = {
            'pid': pid,
            'breathing_condition': breathing_condition,
            'position_condition': position_condition,
            'cadence': cadence,
            'images': flow,
            'labels': labels
        }
    '''
#
    def __init__(self, pids, seq_cap, stack_size, classes, img_folder_name, sessions_obj, body_norm_values = None, norm_values = None, sliding_window = 0, transform=None, body_motor_binary = False, min_max_values = None, body_min_max_values = None, imu_norm_values = None, sessions = None):
        """
        Args:
            pids (list): List of pids to include in the dataset.
            seq_cap (int): a cap on the number of OF images per sequence
            transform (callable, optional): Optional transform to be applied to a sample
        """
        self.vid_index = VideoIndex_Stacked(pids, seq_cap, stack_size, img_folder_name, sessions_obj, window_step = sliding_window, sessions = sessions)
        self.transform = transform
        self.seq_cap = seq_cap
        self.window = sliding_window
        self.stack_size = stack_size
        self.classes = classes
        self.img_folder_name = img_folder_name
        self.norm_values = norm_values
        self.body_norm_values = body_norm_values
        self.body_motor_binary = body_motor_binary
        self.min_max_values = min_max_values
        self.body_mix_max_values = body_min_max_values
        self.imu_norm_values = imu_norm_values
        self.sessions = sessions
        if body_min_max_values != None:
            self.motor_min = body_min_max_values[0]
            self.motor_max = body_min_max_values[1]
        if norm_values != None:
            self.ch1_mean = norm_values[0][0]
            self.ch1_sd = norm_values[0][1]
            self.ch2_mean = norm_values[1][0]
            self.ch2_sd = norm_values[1][1]
        if body_norm_values != None:
            self.motor_mean = body_norm_values[0]
            self.motor_std = body_norm_values[1]
        if min_max_values != None:
            self.ch1_min = min_max_values[0][0]
            self.ch1_max = min_max_values[0][1]
            self.ch2_min = min_max_values[1][0]
            self.ch2_max = min_max_values[1][1]
        if imu_norm_values != None:
            self.lin1_mean = imu_norm_values[0][0][0]
            self.lin1_std = imu_norm_values[0][0][1]
            self.lin2_mean = imu_norm_values[0][1][0]
            self.lin2_std = imu_norm_values[0][1][1]
            self.lin3_mean = imu_norm_values[0][2][0]
            self.lin3_std = imu_norm_values[0][2][1]

            self.gyro1_mean = imu_norm_values[1][0][0]
            self.gyro1_std = imu_norm_values[1][0][1]
            self.gyro2_mean = imu_norm_values[1][1][0]
            self.gyro2_std = imu_norm_values[1][1][1]
            self.gyro3_mean = imu_norm_values[1][2][0]
            self.gyro3_std = imu_norm_values[1][2][1]


    def __len__(self):
        return len(self.vid_index.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        stack = []
        labels = []
        imu_linear = []
        imu_gyro = []
        imu_euler = []
        body_motor = []

        sample_path = self.vid_index.index[idx]
        true_sample = np.load(sample_path)

        last_underscore = sample_path.rfind("_")
        last_period = sample_path.rfind('.')
        true_sample_number = sample_path[last_underscore+1:last_period]
        shortened_sample_path = sample_path[:last_underscore+1] 
        before_hz = len('/data/ommdb/')
        after_hz = sample_path.rindex('Hz/')
        hz = sample_path[before_hz:after_hz]
        before_flow = sample_path.rfind('/flow')
        before_pid = sample_path.find('_')
        if before_flow == -1:
            before_flow = sample_path.rfind('/rgb')
        session_folder_path = sample_path[:before_flow]
        pid = sample_path[before_pid+1:before_pid+4]
        
        motion_h5 = dd.io.load(session_folder_path + '/motion_data_' + str(hz) + 'hz.h5')
        imu_eulers = np.squeeze(motion_h5['imu_euler'])[1:]
        imu_linears = np.squeeze(motion_h5['imu_linear'])[1:]
        imu_gyros = np.squeeze(motion_h5['imu_gyro'])[1:]
        body_motors = np.squeeze(motion_h5['body_motor'])[1:]
        
        for i in range(self.stack_size-1, 0, -1):
            cur_sample_number = int(true_sample_number) - i
            
            if cur_sample_number < 0:
                print(f'whoops for i {i} cur_sample_number {cur_sample_number} and true_sample_number {true_sample_number} at path {sample_path}')
                input('go')
            
            cur_sample_path = shortened_sample_path + str(cur_sample_number) + ".npz"
            cur_sample = np.load(cur_sample_path)
            if 'rgb' in self.img_folder_name:
                if 'img' in cur_sample.files:
                    cur_image = cur_sample['img']
                else:
                    cur_image = cur_sample['flow']
                cur_image = resize(cur_image, (180, 320))
            elif 'flow' in self.img_folder_name:
                #use optical flow
                cur_image = cur_sample['flow']
                if self.min_max_values != None:
                    cur_image_ch1 = cur_image[:,:,0] #channel is at end
                    cur_image_ch2 = cur_image[:,:,1]
                    normed_ch1 = (cur_image_ch1 - self.ch1_min)/(self.ch1_max - self.ch1_min)
                    normed_ch2 = (cur_image_ch2 - self.ch2_min)/(self.ch2_max - self.ch2_min)
                    cur_image = np.stack((normed_ch1, normed_ch2), axis = 2)

                if self.norm_values != None:
                    cur_image_ch1 = cur_image[:,:,0] 
                    cur_image_ch2 = cur_image[:,:,1]
                    normed_ch1 = (cur_image_ch1 - self.ch1_mean)/self.ch1_sd
                    normed_ch2 = (cur_image_ch2 - self.ch2_mean)/self.ch2_sd
                    cur_image = np.stack((normed_ch1, normed_ch2), axis = 2)
            stack.append(cur_image)
            cur_label = cur_sample['label'][0]
            if self.classes == 3:
                if cur_label == 3 or cur_label == str(3):
                    cur_label = 0
            labels.append(cur_label)

            #add in the motion data points!
            cur_body_motor = body_motors[cur_sample_number]
            cur_imu_linear = imu_linears[cur_sample_number]
            cur_imu_euler = imu_eulers[cur_sample_number]
            cur_imu_gyro = imu_gyros[cur_sample_number]
            imu_linear.append(cur_imu_linear)
            imu_euler.append(cur_imu_euler)
            imu_gyro.append(cur_imu_gyro)
            body_motor.append(cur_body_motor)

        
        if 'rgb' in self.img_folder_name:
            #use rgb
            if 'img' in true_sample.files:
                true_image = true_sample['img']
            else:
                true_image = true_sample['flow']
            true_image = resize(true_image, (180, 320))
        elif 'flow' in self.img_folder_name:
            #use optical flow
            true_image = true_sample['flow']
            if self.norm_values != None:
                true_image_ch1 = true_image[:,:,0] #channel is at end
                true_image_ch2 = true_image[:,:,1]
                normed_ch1 = (true_image_ch1 - self.ch1_mean)/self.ch1_sd
                normed_ch2 = (true_image_ch2 - self.ch2_mean)/self.ch2_sd
                true_image = np.stack((normed_ch1, normed_ch2), axis = 2)
        stack.append(true_image)
        true_label = true_sample['label'][0]
        if self.classes == 3:
            if true_label == 3 or true_label == str(3):
                true_label = 0
        labels.append(true_label)

        #add true sample motion data
        true_sample_number = int(true_sample_number)
        true_body_motor = body_motors[true_sample_number]
        true_imu_linear = imu_linears[true_sample_number]
        true_imu_euler = imu_eulers[true_sample_number]
        true_imu_gyro = imu_gyros[true_sample_number]

        imu_linear.append(true_imu_linear)
        imu_euler.append(true_imu_euler)
        imu_gyro.append(true_imu_gyro)
        body_motor.append(true_body_motor)

        #turn to np arrays
        stack = np.array(stack)
        labels = np.array(labels)
        imu_linear = np.array(imu_linear)
        imu_euler = np.array(imu_euler)
        imu_gyro = np.array(imu_gyro)


        if self.body_motor_binary == True:
            #take shape of body_motor and turn it into binary -- 1 if bodybreathing 0 if notbreathing
            if "bodybreathing" in session_folder_path:
                body_motor = np.ones(len(body_motor))
            else:
                body_motor = np.zeros(len(body_motor))
        else:
            if self.body_norm_values != None:
                body_motor = (body_motor - self.motor_mean)/self.motor_std
            elif self.body_mix_max_values != None:
                body_motor = (body_motor - self.motor_min)/(self.motor_max - self.motor_min)
                
            body_motor = np.array(body_motor)

        if self.imu_norm_values != None:
            #normalize the imu 
            imu_linear[:,0] = (imu_linear[:,0] - self.lin1_mean)/self.lin1_std
            imu_linear[:,1] = (imu_linear[:,1] - self.lin2_mean)/self.lin2_std
            imu_linear[:,2] = (imu_linear[:,2] - self.lin3_mean)/self.lin3_std
            imu_gyro[:,0] = (imu_gyro[:,0] - self.gyro1_mean)/self.gyro1_std
            imu_gyro[:,1] = (imu_gyro[:,1] - self.gyro2_mean)/self.gyro2_std
            imu_gyro[:,2] = (imu_gyro[:,2] - self.gyro3_mean)/self.gyro3_std        
        
        '''
        0 = bottom hold
        1 = inhale
        2 = exhale
        3 = top hold
        '''
        
        if self.transform:
            stack = self.transform(stack)

        stack = np.moveaxis(stack, 3, 1) 
        return (stack, labels, len(stack), imu_linear, imu_gyro, imu_euler, body_motor)
    

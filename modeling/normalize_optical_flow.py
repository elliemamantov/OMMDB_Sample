from __future__ import print_function, division
import os
import numpy as np


def get_of_norm_values(to_pids_file_path, pids, img_folder_name):
    ch1_values = []
    ch2_values = []
    
    for session in os.listdir(to_pids_file_path):
        directory_path = to_pids_file_path + "/" + session
        file_path = directory_path + "/" + img_folder_name + "/"
        samples = os.listdir(file_path)
        sample_path = samples[0]
        last_underscore = sample_path.rfind("_")
        last_period = sample_path.rfind('.')
        img_name_header = sample_path[:last_underscore+1]
        num_samples = len(samples)
        for i in range(0, num_samples):
            sample_path = file_path + img_name_header + str(i) + '.npz'
            sample = np.load(sample_path)
            img = sample['flow']                
            ch1 = img[:,:,0]
            ch2 = img[:,:,1]
            ch1_values.append(ch1)
            ch2_values.append(ch2)
        
    assert len(ch1_values) == len(ch2_values)
    ch1_mean = np.mean(ch1_values)
    ch1_sd = np.std(ch1_values)
    ch2_mean = np.mean(ch2_values)
    ch2_sd =  np.std(ch2_values)    
    return (ch1_mean, ch1_sd), (ch2_mean, ch2_sd)


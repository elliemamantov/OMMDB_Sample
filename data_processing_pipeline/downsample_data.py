import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
import deepdish as dd

'''
ARGS PARSING
'''
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folders', type = str, default = None, nargs = '+', help='session folder names you want to process')
parser.add_argument('-o', '--omit', type = str, default = None, nargs = '+', help='folder names to omit from the list created by the root')
parser.add_argument('-r', '--root', type = str, default = None, help='file path for root for session folders')
parser.add_argument('-sr', '--saveRoot', type = str, help = 'separate folder to save downsampled data to', default = None, required = True)
parser.add_argument('-hz', '--hz', type = int, help = 'herz to downsample to', required = True)

args = parser.parse_args()
root = args.root
hz = args.hz
prefix = str(hz) + 'hz_data'
save_root = args.saveRoot

timestamp = "{:%m_%d_%H_%M}".format(datetime.now())

'''
INTENDED FINAL FILE STRUCTURE
HARD DRIVE
|-- other (stuff, original bag files, processed bag files, etc.)
|-- data_for_modeling
    |-- 10Hz
        |-- pid101_notbreathing...
            |-- respiration_data_10hz.h5
            |-- radar_data_10hz.h5
            |-- images_data_10hz.h5
            |-- motion_data_10hz.h5
        |---pid101_breathing...
        |-- etc. (all other little session folders)
    |-- 15Hz
    |-- etc. (any other downsampling rates)
'''

#check if already a folder for the downsample rate
downsample_folder = save_root + '/' + str(hz) + 'Hz'
if not os.path.isdir(downsample_folder):
    os.mkdir(downsample_folder)
    print('No folder for this Hz rate, making one here:', downsample_folder)
else:
    print(f'{downsample_folder} already exists, using as save folder for data.')

#if no folders specified, use all in the root
if args.folders == None:
    scan = os.scandir(root)
    folders = []
    for f in scan:
        if f.is_dir() and ('UNUSABLE' not in f.name) and ('TEST' not in f.name):
            folders.append(f.name)
    num_folders = len(folders)
    if args.omit:
        folders = [x for x in folders if x not in args.omit] #remove omitted folders
else:
    folders = args.folders
    num_folders = 'n/a'

print(f'found {num_folders} folders in root')
if args.omit:
    print(f'{len(args.omit)} ommited folders: {args.omit}')
print(f'{len(folders)} remaining folders: {folders}')    

'''
HELPER FUNCTIONS
'''
def get_closest_previous_datapoint(values, times, target_timestamp):
    assert len(values) == len(times)
    smallest_difference_timestamp = None
    smallest_difference_value = None
    diff_array = times - target_timestamp #get the differences between the times and the timestamp
    neg_array = [elem for elem in diff_array if elem <= 0] #get rid of positive values -- these are later in time 
    min_diff = max(neg_array) #get the closest by taking the highest value of the neg. difference
    index = np.where(diff_array == min_diff)
    smallest_difference_value = values[index]
    smallest_difference_timestamp = times[index]
    assert len(smallest_difference_value) == 1
    assert len(smallest_difference_timestamp) == 1
    return smallest_difference_timestamp, smallest_difference_value, index

def get_closest_previous_datapoint_OLD(rows, time_index, value_index, target_timestamp):
    smallest_difference = float("inf")
    smallest_difference_timestamp = None
    smallest_difference_value = None

    for i in range(len(rows)):
        row = rows[i]
        time_value = float(row[time_index])
        sensor_value = row[value_index]

        if time_value < target_timestamp: #only check timestamps that are before the target timestamp
            difference = target_timestamp - time_value
            if difference < smallest_difference:
                smallest_difference_timestamp = time_value
                smallest_difference_value = sensor_value

        if time_value > target_timestamp: #we've reached a timestamp greater than the target timestamp, so we can stop searching
            return smallest_difference_timestamp, smallest_difference_value

    return smallest_difference_timestamp, smallest_difference_value

def remove_spaces_and_convert_float(arr):
    num = 0
    values = []
    new_arr = []
    for i, t in enumerate(arr):
        if ' ' in str(t):
            values.append(t)
            t = t.replace(" ", "")
            num += 1
        new_arr.append(float(t))
    return num, values, np.asarray(new_arr)


'''
MAIN PROCESSING
'''
#save any skipped folders and why
no_labels = []

for f in folders:
    print(f'~~~~~~~~~~~~ {f} ~~~~~~~~~~~~')
    path = root + '/' + f

    # Open all the CSVs and turn into pandas dfs 
    print('loading csvs')
    resp_belt_path = path +"/resp_belt.csv" #contains user label as well as target
    imu_path = path + "/imu.csv"
    motors_path = path + "/motors.csv"
    radar_path =  path + '/radar_raw.csv'
    images_path = path + '/raspicam_node-image-compressed.csv'

    resp_belt_df = pd.read_csv(resp_belt_path, sep=',')
    imu_df = pd.read_csv(imu_path, sep=',')
    motors_df = pd.read_csv(motors_path, sep=',')
    radar_df = pd.read_csv(radar_path, sep=',')
    #print('starting to load images csv...')

    images_df = pd.read_csv(images_path, sep=',', usecols=["Time"],)
    print('done')

    #check if has labeles in resp_data, otherwise skip and note which ones:
    if 'label' in resp_belt_df.columns:
        resp_data = resp_belt_df['label'].to_numpy()
    else:
        no_labels.append(f)
        print('no labels in resp_belt.csv! moving on...')
        continue
    
    #check if has targets from motors
    if 'Target' in resp_belt_df.columns:
        has_targets = True
        resp_targets = resp_belt_df['Target'].to_numpy()
    else:
        has_targets = False
    
    resp_times = resp_belt_df['Time'].to_numpy()
    body_values = motors_df['body_pos'].to_numpy()
    body_times = motors_df['body_times'].to_numpy()
    head_values = motors_df['head_pos'].to_numpy()
    head_times = motors_df['head_times'].to_numpy()
    euler0_clean = imu_df['euler_0'].to_numpy()
    euler1_clean = imu_df['euler_1'].to_numpy()
    euler2_clean = imu_df['euler_2'].to_numpy()
    linear0_clean = imu_df['linear_0'].to_numpy()
    linear1_clean = imu_df['linear_1'].to_numpy()
    linear2_clean = imu_df['linear_2'].to_numpy()
    gyro0_clean = imu_df['gyro_0'].to_numpy()
    gyro1_clean = imu_df['gyro_1'].to_numpy()
    gyro2_clean = imu_df['gyro_2'].to_numpy()
    imu_times = imu_df['Time'].to_numpy()

    #a wee bit of processing to get the radar raw values as arrays from the csv!
    cols = radar_df.columns
    real_ix_start = np.where(cols == 'real_0')[0][0]
    imag_ix_start = np.where(cols == 'imaginary_0')[0][0]
    real_ix_end = imag_ix_start
    radar_real = radar_df[cols[real_ix_start:real_ix_end]].to_numpy()
    radar_imag = radar_df[cols[imag_ix_start:]].to_numpy()
    assert np.shape(radar_real) == np.shape(radar_imag)
    radar_times = radar_df['Time'].to_numpy()



    print('loading images npy')
    images = np.load(path+'/images.npy') # TODO: add the images stuff!
    print('imapges.npy shape', images.shape)
    images_times = images_df['Time'].to_numpy()
    print('done')
    
    times_arrs = [resp_times, body_times, head_times, imu_times, radar_times, images_times]    
    times_arrs_names = ['resp_times', 'body_times', 'head_times', 'imu_times', 'radar_times', 'images_times'] 
    new_times_arrs = []
    for i, arr in enumerate(times_arrs):
        num, values, new_arr = remove_spaces_and_convert_float(arr)
        new_times_arrs.append(new_arr)
        print(f'removed {num} spaces from {times_arrs_names[i]}')
    
        resp_times, body_times, head_times, imu_times, radar_times, images_times = new_times_arrs[0], new_times_arrs[1], new_times_arrs[2], new_times_arrs[3], new_times_arrs[4], new_times_arrs[5]

  
    #Get each sensor's start time
    imu_start_time = float(imu_times[0])
    head_motors_start_time = float(body_times[0])
    body_motors_start_time = float(head_times[0])
    radar_start_time = float(radar_times[0])
    resp_start_time = resp_times[0]
    images_start_time = float(images_times[0])


    #Get each sensor's end time
    imu_end_time = float(imu_times[-1])
    head_motors_end_time = float(body_times[-1])
    body_motors_end_time = float(head_times[-1])
    radar_end_time = float(radar_times[-1])
    resp_end_time = resp_times[-1]
    images_end_time = float(images_times[-1])
    
    #Get the latest start time
    start_times = [imu_start_time, head_motors_start_time, body_motors_start_time, resp_start_time, radar_start_time, images_start_time]
    last_start_time = float(max(start_times))

    #get the last time
    end_times = [imu_end_time, head_motors_end_time, body_motors_end_time, resp_end_time, radar_start_time images_end_time]
    last_end_time = float(max(end_times))

    #Make the target timestamps list
    target_timestamps = []
    time_between_points = 1/hz
    for i in range(hz*90): #90 seconds of timestamps
        time = last_start_time + i*time_between_points
        if time > last_end_time:
            break
        else:
            target_timestamps.append(last_start_time + i*time_between_points)

    #Process all the data
    head_motors_downsampled = []
    head_motors_downsampled_times = []
    body_motors_downsampled = []
    body_motors_downsampled_times = []

    imu_gyro_downsampled = []
    imu_linear_downsampled = []
    imu_euler_downsampled = []
    imu_downsampled_times = []

    radar_real_downsampled = []
    radar_imaginary_downsampled = []
    radar_downsampled_times = []

    resp_data_downsampled = [] #THESE ARE THE LABLES
    resp_targets_downsampled = [] #THESE ARE THE TARGETS FROM THE ROBOT BREATHING (WHAT USERS SHOULD MATCH TO)
    resp_downsampled_times = []


    images_downsampled = []
    images_downsampled_times = []

    last_resp_index = [-1000]
    for i, t in enumerate(target_timestamps):
        closest_resp_data_time, closest_resp_data_value, resp_index = get_closest_previous_datapoint(resp_data, resp_times, t)
        if resp_index[0] == last_resp_index[0]:
            print('same index for frame: ', i)
        last_resp_index = resp_index
        
        
        resp_data_downsampled.append(closest_resp_data_value)
        if has_targets:
            closest_resp_target_time, closest_target_data_value, target_index = get_closest_previous_datapoint(resp_targets, resp_times, t)
            resp_targets_downsampled.append(closest_target_data_value) 
            assert closest_resp_data_time == closest_resp_target_time
        resp_downsampled_times.append(closest_resp_data_time)
        
        #process motors
        closest_body_time, closest_body_value, body_index = get_closest_previous_datapoint(body_values, body_times, t)
        body_motors_downsampled.append(closest_body_value)
        body_motors_downsampled_times.append(closest_body_time)
        
        closest_head_time, closest_head_value, head_index = get_closest_previous_datapoint(head_values, head_times, t)
        head_motors_downsampled.append(closest_head_value)
        head_motors_downsampled_times.append(closest_head_time)
        
        #process radar
        closest_real_time, closest_real_value, real_index = get_closest_previous_datapoint(radar_real, radar_times, t)
        closest_imaginary_time, closest_imaginary_value, imag_index = get_closest_previous_datapoint(radar_imag, radar_times, t)
        radar_real_downsampled.append(closest_real_value)
        radar_imaginary_downsampled.append(closest_imaginary_value)
        assert closest_real_time == closest_imaginary_time
        radar_downsampled_times.append(closest_real_time)

        #process imu
        eulers = [euler0_clean, euler1_clean, euler2_clean]
        linears = [linear0_clean, linear1_clean, linear2_clean]
        gyros = [gyro0_clean, gyro1_clean, gyro2_clean]
        imus = [eulers, linears, gyros]
        imus_names = ['eulers', 'linears', 'gyros']
        for i, m in enumerate(imus):
            value = ()
            for dim in m:
                closest_time, closest_value, indx = get_closest_previous_datapoint(dim, imu_times, t)
                value = value + (closest_value[0],)
            if imus_names[i] == 'eulers':
                imu_euler_downsampled.append(value)
            elif imus_names[i] == 'linears':
                imu_linear_downsampled.append(value)
            elif imus_names[i] == 'gyros':
                imu_gyro_downsampled.append(value)
                imu_downsampled_times.append(closest_time)

        closest_img_time, closest_img_value, img_index = get_closest_previous_datapoint(images, images_times, t)
        images_downsampled.append(closest_img_value)  
        images_downsampled_times.append(closest_img_time)

    #convert to np arrays
    head_motors_downsampled = np.asarray(head_motors_downsampled)
    head_motors_downsampled_times = np.asarray(head_motors_downsampled_times)
    body_motors_downsampled = np.asarray(body_motors_downsampled)
    body_motors_downsampled_times = np.asarray(body_motors_downsampled_times)

    imu_gyro_downsampled = np.asarray(imu_gyro_downsampled)
    imu_linear_downsampled = np.asarray(imu_linear_downsampled)
    imu_euler_downsampled = np.asarray(imu_euler_downsampled)
    imu_downsampled_times = np.asarray(imu_downsampled_times)

    radar_real_downsampled = np.asarray(radar_real_downsampled)
    radar_imaginary_downsampled = np.asarray(radar_imaginary_downsampled)
    radar_downsampled_times = np.asarray(radar_downsampled_times)

    resp_data_downsampled = np.asarray(resp_data_downsampled)
    if has_targets:
        resp_targets_downsampled = np.asarray(resp_targets_downsampled)
    resp_downsampled_times = np.asarray(resp_downsampled_times)

    images_downsampled = np.asarray(images_downsampled)
    images_downsampled_times = np.asarray(images_downsampled_times)

    
    length = len(target_timestamps)

    assert len(head_motors_downsampled) == length
    assert len(head_motors_downsampled) == length
    assert len(body_motors_downsampled) == length
    assert len(imu_euler_downsampled) == length
    assert len(imu_linear_downsampled) == length
    assert len(imu_gyro_downsampled) == length
    assert len(radar_real_downsampled) == length
    assert len(radar_imaginary_downsampled) == length
    assert len(resp_data_downsampled) == length
    assert len(images_downsampled) == length
    if has_targets:
        assert len(resp_targets_downsampled) == length

       

    #create folder for the session if needed
    session_folder = downsample_folder + '/' + f
    session_folder = session_folder[:25] + 'body' + session_folder[25:]
    if not os.path.isdir(session_folder):
        os.mkdir(session_folder)

    #get metadata about this session
    pid = f[4:7]
    breathing_condition = ""
    position_condition = ""
    cadence = ""
    survey_header = None

    if "not_breathing" in f:
        breathing_condition = "not_breathing"
        survey_header = 'NB '
    else:
        breathing_condition = "breathing"
        survey_header = 'B '

    if "table" in f:
        position_condition = "table"
        survey_header = survey_header + 'T '
    else:
        position_condition = "lap"
        survey_header = survey_header + 'L '

    if "323" in f:
        cadence = "3-2-3"
        survey_header = survey_header + '323'
    elif "4444" in f:
        cadence = "4-4-4-4"
        survey_header = survey_header + '4444'
    else:
        cadence = "5-3-5"
        survey_header = survey_header + '535'

    response = None

    if has_targets:
        respiration_data = {
            'pid': pid,
            'breathing_condition': breathing_condition,
            'position_condition': position_condition,
            'cadence': cadence,
            'hz': hz,
            'resp_labels': resp_data_downsampled,
            'resp_targets': resp_targets_downsampled,
            'resp_times': resp_downsampled_times,
            'survey_response': response
        }
    else:
        respiration_data = {
            'pid': pid,
            'breathing_condition': breathing_condition,
            'position_condition': position_condition,
            'cadence': cadence,
            'hz': hz,
            'resp_labels': resp_data_downsampled,
            'resp_times': resp_downsampled_times,
            'survey_response': response
        }
    dd.io.save(session_folder + '/respiration_data_' + str(hz) + 'hz.h5', respiration_data)


    motion_data = {
        'pid': pid,
        'breathing_condition': breathing_condition,
        'position_condition': position_condition,
        'cadence': cadence,
        'hz': hz,
        'imu_euler': imu_euler_downsampled,
        'imu_linear': imu_linear_downsampled,
        'imu_gyro': imu_gyro_downsampled,
        'imu_times': imu_downsampled_times,
        'body_motor': body_motors_downsampled,
        'body_motor_times': body_motors_downsampled_times,
        'head_motor': head_motors_downsampled,
        'head_motor_times': head_motors_downsampled_times
    }
    dd.io.save(session_folder + '/motion_data_' + str(hz) + 'hz.h5', motion_data)

    
    image_data = {
        'pid': pid,
        'breathing_condition': breathing_condition,
        'position_condition': position_condition,
        'cadence': cadence,
        'images': images_downsampled,
        'images_times': images_downsampled_times
    }
    dd.io.save(session_folder + '/image_data_' + str(hz) + 'hz.h5', image_data)


    print('all data saved for: ', f)


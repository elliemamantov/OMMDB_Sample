import numpy as np
import argparse
import os
import deepdish as dd
import cv2

#### ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folders', type = str, default = None, nargs = '+', help='folders you want to pull image npzs from')
parser.add_argument('-o', '--omit', type = str, default = None, nargs = '+', help='folder names to omit from the list created by the root')
parser.add_argument('-r', '--root', type = str, default = None, help='file path for root for folders you want')
parser.add_argument('-hz', '--hertz', type = int, default = 10, help='downsample rate to pull from')
parser.add_argument('-h5', '--h5', dest='make_h5', action='store_true')
parser.add_argument('-im', '--images', dest='make_images', action='store_true')
args = parser.parse_args()
root = args.root
hz = args.hertz
make_h5 = args.make_h5
make_folder = args.make_images



#if no folders specified, use all in the root
if args.folders == None:
    scan = os.scandir(root)
    folders = []
    for f in scan:
        if f.is_dir():
            if ('lap' in f.name) or ('table' in f.name):
                folders.append(f.name)
    num_folders = len(folders)
    if args.omit != None:
        folders = [x for x in folders if x not in args.omit] #remove omitted folders
else:
    folders = args.folders
    num_folders = 'n/a'

print(f'found {num_folders} folders in root for downsample rate {hz} hz')
if args.omit:
    print(f'{len(args.omit)} ommited folders: {args.omit}')
print(f'{len(folders)} remaining folders: {folders}')

def show_flow_sample(session, flow_data, i):
    # get random i for smaple
    flow1 = flow_data[i]
    flow2 = flow_data[i+1]
    flow3 = flow_data[i+2]

    # Computes the magnitude and angle of the 2D vectors
    magnitude1, angle1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
    magnitude2, angle2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
    magnitude3, angle3 = cv2.cartToPolar(flow3[..., 0], flow3[..., 1])

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask1 = np.zeros((flow1.shape[0], flow1.shape[1], 3), dtype = np.float32)
    mask2 = np.zeros((flow2.shape[0], flow2.shape[1], 3), dtype = np.float32)
    mask3 = np.zeros((flow3.shape[0], flow3.shape[1], 3), dtype = np.float32)
    
    # Sets image saturation to maximum
    mask1[..., 1] = 255
    mask2[..., 1] = 255
    mask3[..., 1] = 255
      
    # Sets image hue according to the optical flow direction
    mask1[..., 0] = angle1 * 180 / np.pi / 2
    mask2[..., 0] = angle2 * 180 / np.pi / 2
    mask3[..., 0] = angle3 * 180 / np.pi / 2
      
    # Sets image value according to the optical flow magnitude (normalized)
    mask1[..., 2] = cv2.normalize(magnitude1, None, 0, 255, cv2.NORM_MINMAX)
    mask2[..., 2] = cv2.normalize(magnitude2, None, 0, 255, cv2.NORM_MINMAX)
    mask3[..., 2] = cv2.normalize(magnitude3, None, 0, 255, cv2.NORM_MINMAX)
      
    # Converts HSV to RGB (BGR) color representation
    rgb1 = cv2.cvtColor(mask1, cv2.COLOR_HSV2BGR)
    rgb2 = cv2.cvtColor(mask2, cv2.COLOR_HSV2BGR)
    rgb3 = cv2.cvtColor(mask3, cv2.COLOR_HSV2BGR)
      
    # Opens a new window and displays the output frame
    hori = np.concatenate((rgb1, rgb2, rgb3), axis=1)


def get_optical_flow(session, imgs):
    print('starting optical flow conversion')
    flow_data = []
    for i in range(len(imgs)-1):
        #for each timestep of video
        first_im = imgs[i]
        second_im = imgs[i+1]
        prvs = cv2.cvtColor(first_im, cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(second_im, cv2.COLOR_BGR2GRAY)        
        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_data.append(flow)

    flow_data = np.asarray(flow_data)
    return flow_data

#takes in h5 file with downsampled images and adds in optical flow in between them
for f in folders:
    path = root + '/' + f
    h5_exists = os.path.exists(path + '/flow_data_' + str(hz) + 'hz.h5') #flow_data_10Hz.h5
    folder_exists = os.path.exists(path + '/flow_images') or os.path.exists(path + '/flow_images_float_half') or os.path.exists(path + '/flow_images_float')#flow folder of optical flow images
    file_name = 'image_data_' + str(hz) + 'hz.h5'
    imgs_data = dd.io.load(path + '/' + file_name)
    imgs = imgs_data['images']
    imgs = imgs.squeeze()

    #get the labels
    file_name = 'respiration_data_' + str(hz) + 'hz.h5'
    resp_h5 = dd.io.load(path + '/' + file_name)
    if 'resp_data' in resp_h5.keys():
        labels = resp_h5['resp_data']
    elif 'resp_labels' in resp_h5.keys():
        labels = resp_h5['resp_labels']

    #run the optical flow bits in float32
    flow_float = get_optical_flow(f, imgs)

    assert len(flow_float) == (len(labels)-1)
    

    #save into the h5 file
    if make_h5 == True:
        print('saving h5 file now')
        pid = imgs_data['pid']
        breathing_condition = imgs_data['breathing_condition']
        position_condition = imgs_data['position_condition']
        cadence = imgs_data['cadence']

        flow_data = {
            'pid': pid,
            'breathing_condition': breathing_condition,
            'position_condition': position_condition,
            'cadence': cadence,
            'images': flow,
            'labels': labels[1:] #taking the second frame as the label for each OF image
        }
        
        save_name = path + '/flow_data_' + str(hz) + 'hz.h5'
        dd.io.save(save_name, flow_data)
        print('h5 saved to', save_name)

    #save the images and labels into npys, using the label as the second frame in the optical flow pair
    if make_folder:
        flow_float_folder = path + '/flow_images_float'
        flow_float_folder_small = path + '/flow_images_float_half'
        
        #1/2 SIZED FLAOAT
        if not os.path.isdir(flow_float_folder_small):
            os.mkdir(flow_float_folder_small)
            print('No folder for optical flow float images for this session, making one here:', flow_float_folder_small)
        else:
            print(f'{flow_float_folder_small} already exists, using as folders for ')

        
        for i, frame in enumerate(flow_float):
            #save smaller images
            scale = .5
            dim = (int(frame.shape[1]*scale), int(frame.shape[0]*scale))
            resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # dim = (WIDTH, height)
            frame_name = flow_float_folder_small + '/' + f +'_OF_' + str(i) + '.npz'
            np.savez(frame_name, flow=resized, label=labels[i+1])




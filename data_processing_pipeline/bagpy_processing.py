import bagpy
from bagpy import bagreader
import numpy as np
import os
import argparse

'''
ARGS PARSING
'''
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', type = str, help='root file path for bags -- bag files then just need to be session_name.bag', default = None)
parser.add_argument('-b', '--bagFiles', type = str, default = None, nargs='+', help='bag file paths (can be multiple). If root is included in args, no need full path. If not, ensure bag files have full path.')
parser.add_argument('-o', '--omit', type = str, default = None, nargs = '+', help='folder names to omit from the list created by the root')

args = parser.parse_args()
root = args.root
bags = args.bagFiles
save = args.saveRaw

#add .bag to omits
if args.omit:
    omits = args.omit
    omits = [folder + '.bag' for folder in omits]

#if no folders specified, use all in the root
if args.bagFiles == None:
    print('no specific bags set, collecting from root')
    scan = os.scandir(root)
    bags = []
    for f in scan:
        if f.is_dir() and ('TEST' not in f.name):
            bags.append(f.name + '.bag')
    if args.omit != None:
        bags = [x for x in bags if x not in omits] #remove omitted folders
    print('bags in root (after omissions):', bags)
else:
    bags = args.bagFiles

if root != None:
    for i, name in enumerate(bags):
        bags[i] = root + '/' + name

def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

'''
SETUP BAG READING
'''
for b in bags:
    print('Processing Bag:', b[-30:])
    bag = bagreader(b)
    
    csvfiles = []
    topics = ['/raspicam_node/image/compressed', '/imu', '/radar_config', '/radar_raw', '/resp_belt', '/dynamixel_workbench_controllers/dynamixel_state']
    for t in topics:
        data = bag.message_by_topic(t)
        csvfiles.append(data)
    

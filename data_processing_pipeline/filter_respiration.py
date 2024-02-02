from scipy.ndimage import gaussian_filter1d
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

#### ARGS PARSING
parser = argparse.ArgumentParser()
parser.add_argument('-fp', '--filepaths', type = str, nargs = '+', help='file path for csvs you want if solo, the bag folder names if with a root')
parser.add_argument('-s', '--sigma', type = int, default = 3, help='sigma for guassian')
parser.add_argument('-r', '--root', type = str, default = None, help='file path for root for bags you want')
parser.add_argument('-as', '--alsoSave', type = str, help = 'separate folder to save csvs to', default = None)

args = parser.parse_args()
file_paths = args.filepaths
sigma = args.sigma
root = args.root
window_size = args.windowSize
if args.alsoSave != None:
    alt_save = root + '/' + args.alsoSave
    if not os.path.isdir(alt_save):
        os.mkdir(alt_save)

if root != None:
    for file in file_paths:
        os.chdir(root)
        session = file
        file = str(file+'/resp_belt.csv')
        df = pd.read_csv(file, sep=',')
        force = df['float.data']
        times = df['Time']
        dx = 1
        slopes = np.gradient(force, dx)
        second_der = np.gradient(slopes, dx)
        
        
        df['slope_der'] = second_der*100
        df['slope'] = slopes*100
        df['filtered_5'] = gaussian_filter1d(force, 5)
        df['filtered_10'] = gaussian_filter1d(force, 10)
        df['filtered_15'] = gaussian_filter1d(force, 15)
        df['slopes_new'] = get_all_slopes(force, times, window_size)
        df['slopes_gauss'] = get_all_slopes(gaussian_filter1d(force, 10), times, window_size)
        if args.alsoSave != None:
            df.to_csv(alt_save+'/resp_belt_filtered_'+session+'.csv')
            print('alt saved csv to:', alt_save+'/resp_bel_filtered_'+session+'.csv')
        df.to_csv(file[:-4]+'_filtered_'+session+'.csv')
else:
    for file in file_paths:
        session = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file, sep=',')
        force = df['float.data']
        times = df['Time']
        dx = 1
        slopes = np.gradient(force, dx)
        second_der = np.gradient(slopes, dx)
    
        
        df['slope_der'] = second_der*100
        df['slope'] = slopes*100
        df['filtered_5'] = gaussian_filter1d(force, 5)
        df['filtered_10'] = gaussian_filter1d(force, 10)
        df['filtered_15'] = gaussian_filter1d(force, 15)
        df['slopes_new'] = get_all_slopes(force, times, window_size)
        df['slopes_gauss'] = get_all_slopes(gaussian_filter1d(force, 10), times, window_size)
        df.to_csv(file[:-4]+'_filtered_'+session+'.csv')
        if args.alsoSave != None:
            df.to_csv(alt_save+'/resp_belt_filtered_'+session+'.csv')
            print('alt saved csv to:', alt_save+'/resp_bel_filtered_'+session+'.csv')

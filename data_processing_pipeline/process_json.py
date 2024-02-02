import json
import os
import argparse
import csv

#### ARGS PARSING

parser = argparse.ArgumentParser()
parser.add_argument('-fp', '--filepaths', type = str, nargs = '+', default = None, help='file path for bags you want')
parser.add_argument('-g', '--gap', type = float, default = .01, help='gap in seconds that indicates small gaps between annotated labels')
parser.add_argument('-a', '--annotator', type = str, default = None, help='annotator name')
parser.add_argument('-sr', '--saveRoot', type = str, default = None, help='path to folder to save annotations, if not in cwd')
parser.add_argument('-sf', '--saveFolder', type = str, default = None, help='folder name to save annotations')
parser.add_argument('-br', '--bagRoot', required = True, type = str, default = None, help='root to bagfiles folder, to save the resp data into session folders')
args = parser.parse_args()
max_gap = args.gap
anno = args.annotator
save_root = args.saveRoot
save_folder = args.saveFolder+'_'+anno

if args.filepaths != None:
    file_paths = args.filepaths
else: 
    print("No files provided!")

if args.saveRoot == None:
    save_root = os.getcwd() + '/'
else:
    save_root = save_root + '/'

if os.path.isdir(save_root+'/'+save_folder):
    p = save_root+'/'+save_folder
    print(f"{p} is a already a directory")
else:
    print('no annotations folder, creating now')
    os.mkdir(save_folder)

if bag_root != None:
    print('saving processed resp csvs into individual session folders at: ', bag_root)

'''
Helper Functions
'''
def get_short(label):
    short_label = -1
    if label == "Inhale":
        short_label = 1
    elif label == "Exhale":
        short_label = 2
    elif label == "Top Hold":
        short_label = 3
    elif label == "Bottom Hold":
        short_label = 0
    elif label == "Deviance":
        short_label = -100
    else:
        print('\033[91m' + "ERROR: Unknown label name" + '\x1b[0m')
    return short_label



def get_gap_label(prev, nxt):
    if prev == "Bottom Hold" and nxt == "Top Hold":
        return 1 #Inhale
    elif prev == "Inhale" and nxt == "Exhale":
        return 3 #Top Hold
    elif prev == "Top Hold" and nxt == "Bottom Hold":
        return 2 #Exhale
    elif prev == "Exhale" and nxt == "Inhale":
        return 0 #Bottom Hold
    else:
        print('\033[91m' + "ERROR: prev is", prev, "next is", nxt + '\x1b[0m')

def clean_labels(labels, dev_csv_name):
    ##add in csv saving
    print(f'found {len(labels)} labels')
    err_count = 0
    dev_count = 0
    null_count = 0
    to_remove = []
    f3 = open(save_root+save_folder+'/'+dev_csv_name, 'w')
    dev_writer = csv.writer(f3)
    dev_writer.writerow(["", "begin_time", "end_time", "duration", "label"])

    for label in labels:
        if 'start' not in label:
            to_remove.append(label)
            err_count = err_count + 1
        elif label['end'] == None or label['end'] == "null":
            to_remove.append(label)
            null_count = null_count + 1
        elif label['timeserieslabels'][0] == "Deviance":
            dev_count = dev_count + 1
            to_remove.append(label)
            dev_writer.writerow(['tier', str(label['start']), str(label['end']), str(label['end'] - label['start']), get_short(label['timeserieslabels'][0])])

    for rem in to_remove:
        labels.remove(rem)
        
    print(f'Removed {err_count} label(s) due to missing start time!')
    print(f'Removed {null_count} label(s) due to null end time!')
    print(f'Removed {dev_count} deviances!')
    print(f'Remaining labels: {len(labels)}')
    f3.close()
    return labels

def get_first_label(labels):
    earliest_time = float("inf")
    earliest_label = ""
    for label in labels:
        if 'start' not in label:
            print('no start time indicated in json')
            break
        start_time = label['start']
        if start_time < earliest_time:
            earliest_time = start_time
            earliest_label = label
    return earliest_label



def get_next_label(labels, prev_start_time, prev_label):
    closest_time_diff = float("inf")
    closest_label = ""

    for label in labels:

        if 'start' not in label:
            break

        start_time = label['start']
    
        if start_time == prev_start_time and label != prev_label:
            print('found a same start time')
            print('label 1:', label)
            print('label 2:', prev_label)
        
        if start_time > prev_start_time:
            diff = start_time - prev_start_time

            if diff < closest_time_diff:
                closest_time_diff = diff
                closest_label = label
    return closest_label

for json_file_name in file_paths:

    deviances = []
    f = open(json_file_name)

    all_data = json.load(f)

    for data in all_data:
        name = data['timeseries']
        print('\033[96m'+'now processing', name + '\x1b[0m')
        index = name.find("resp")
        new_csv_file_name = anno + '_' + name[index:-4] +"_labels.csv"
        new_dev_csv_file_name = anno + '_' + name[index:-4] +"_labels_DEVIANCES.csv"
        session_i = name.find("pid")
        session_name = name[session_i:-4]
        num_underscores = session_name.count('_')
        if num_underscores == 5:
            underscore_i = session_name.rfind('_')
            session_name = session_name[:underscore_i]
        f2 = open(save_root+save_folder+'/'+new_csv_file_name, 'w')
        writer = csv.writer(f2)
        writer.writerow(["", "begin_time", "end_time", "duration", "label"])

        #if also saving in the bag folders...
        if bag_root != None:
            f4 = open(bag_root+'/'+session_name+'/resp_labes.csv', 'w')
            writer4 = csv.writer(f4)
            writer4.writerow(["", "begin_time", "end_time", "duration", "label"])

        labels = data['label']
        labels = clean_labels(labels, new_dev_csv_file_name)
        label = get_first_label(labels)
        start = label['start']
        true_start = start
        end = label['end']
        label_label = label['timeserieslabels'][0]
        short_label = get_short(label_label)
        duration = end - start
        tier = "tier"
        row = [tier, str(start), str(end), str(duration), short_label]
        writer.writerow(row)
        if bag_root != None:
            writer4.writerow(row)
        prev_label = label
        for i in range(1,len(labels)):
            label = get_next_label(labels, true_start, prev_label)
            prev_end = prev_label['end']
            prev_label_label = prev_label['timeserieslabels'][0]
            start = label['start']
            true_start = start
            end = label['end']
            label_label = label['timeserieslabels'][0]
            if label_label == "Deviance":
                print('found a deviance that got through cleaning!')
            else:
                short_label = get_short(label_label)
                duration = end - start
                #if we have an overlapping area, make the start of this section the end of the last section
                if start < prev_end: 
                    start=prev_end

                #if we have a gap between the last and the current, add the correct section
                if start > prev_end:
                    diff = start - prev_end
                    if diff < max_gap:
                        print(f'mini gap detected between {prev_label_label} and {label_label} with difference of {diff}, adjusting start of phase')
                        start=prev_end
                    else:
                        print(f'large gap detected between {prev_label_label} and {label_label} with difference of {diff}, treating like a new phase')
                        gap_short_label = get_gap_label(prev_label_label, label_label)
                        gap_start = prev_end
                        gap_end = start
                        gap_duration = gap_end- gap_start
                        gap_row = [tier, gap_start, gap_end, gap_duration, gap_short_label]
                        writer.writerow(gap_row)
                        if bag_root != None:
                            writer4.writerow(gap_row)

                row = [tier, start, end, duration, short_label]
                writer.writerow(row)
                if bag_root != None:
                    writer4.writerow(row)

                prev_label = label
        f2.close()
        if bag_root != None:
            f4.close()
    print('num sessions found:', len(all_data))
    

    f.close()




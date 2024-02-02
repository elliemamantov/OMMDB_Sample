'''
region - Libraries
'''
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_video
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
from datetime import datetime
import argparse

from tqdm import tqdm 
import tracemalloc
import gc

from torch.utils.tensorboard import SummaryWriter

from lstm_dataset import OF_Dataset_sequenced_sliding
from optical_flow_lstm_models_archived import FullModel_both, IMUNet, FullModel_imu, FullModel_no_motion, FullModel_body_motion, CNN_simple, CNN_2, LSTM
from normalize_optical_flow import get_of_norm_values
import smtplib
import string
import random
from get_pids import sessionsClass
from normalize_motor import get_motor_norm_params
from crossval_info import get_crossval5_of_norm, get_crossval5_pids, get_crossval8_sets
'''
end region - Libraries
'''

'''
region - Args parsing and variables setup
'''
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-lr", "--learning_rate", default = .01, type = float, help = "learning rate")
parser.add_argument("-e", "--numEpochs", default = 3, type = int, help = "number of epochs")
parser.add_argument("-bs", "--batchSize", default = 2, type = int, help = "batch size")
parser.add_argument("-h", "--imgHeight", default = 180, type = int)
parser.add_argument("-w", "--imgWidth", default = 320, type = int)
parser.add_argument("-ic", "--inChannels", default = 2, type = int)
parser.add_argument("-oc", "--outChannels", default = 1, type = int)
parser.add_argument("-lstmh", "--lstmHiddenSize", default = 3, type = int)
parser.add_argument("-k", "--kernelSize", default = 5, type = int)
parser.add_argument("-lstml", "--lstmLayers", default = 2, type = int)
parser.add_argument("-cnnl", "--cnnLayers", default = 3, type = int)
parser.add_argument("-cf", "--convFeatures", default = 100, type = int)
parser.add_argument('if', '--IMUFeatures'), default = 10, type = int)
parser.add_argument("-tf", "--testFiles", nargs='+', type=str, default = None, help = 'adding specific files to test with')
parser.add_argument("-fr", "--filesRoot", default = './', type = str, help = 'root path of the npz files to train on')
parser.add_argument("-cp", "--convDropout", type=float, default = 0, help = 'convolutional dropout')
parser.add_argument("-lp", "--lstmDropout", type=float, default = 0, help = 'lstm dropout')
parser.add_argument("-l2", "--l2Reg", type=float, default = 0, help = 'L2 Regularization')
parser.add_argument("-sc", "--seqCap", type=int, default = None, help = 'cap the number of frames per video')
parser.add_argument("-dl", "--debouncingLimit", type=int, default = 10, help = 'cap the number of times your val loss increases')
parser.add_argument("-p", "--prefix", type=str, default = '', help = 'prefix like d10')
parser.add_argument("-l", "--label", type = str, default = '', help = 'extra label for tensorboard')
parser.add_argument("-nw", "--numWorkers", default = 2, type = int, help = "number of workers for data loaders") #increase to like 100 or 1024
parser.add_argument("-st", "--stackSize", required = True, type = int, help = "number of OF frames in stack")
parser.add_argument("-cl", "--classes", type = int, default = 3, help = "number of classes to predict")
parser.add_argument('-sv', '--saveModel', dest='saveModel', action='store_true')
parser.add_argument("-gpus", "--gpus", default = [0], type = int, nargs = '+', help = "which gpus to parallelize over")
parser.add_argument("-sw", "--slidingWindow", type = int, default = 0, help = "number of OF frames to skip when sliding the window of frames to stack")
parser.add_argument("-in", "--imgName", type = str, default = 'flow_images_float_half', help = 'image folder name like flow_images_float_half or rgb_images_full')
parser.add_argument("-sh", "--shoulderStates", required = True, type = str, nargs = '+', help = "good, medium, bad, missing etc. ")
parser.add_argument("-br", "--breathingStates", required = True, type = str, nargs = '+', help = "notbreathing, bodybreathing, breathing")
parser.add_argument("-po", "--positionStates", required = True, type = str, nargs = '+', help = "table, lap")
parser.add_argument("-mo", "--motionStates", default = 'none', type = str, choices = ['body_motor', 'imu', 'both', 'none', 'linear', 'body_binary'], help = "body_motor, imu, or both")

args = parser.parse_args()

debug_print = False
def dprint(label, value):
    if debug_print == True:
        print(label, value)

#data for saving summary report of model
lines = []

#date time for folder saving
now = datetime.now()
key = res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
dt_string = str(now.strftime("%m_%d_%Y_%H_%M_%S")) + '_' + key
lines.append(dt_string)
lines.append('tensorboard label:' + args.label)

#args parsing
learning_rate = args.learning_rate
lines.append('lr = ' + str(learning_rate))
num_epochs = args.numEpochs
lines.append('lr = ' + str(learning_rate))
batch_size = args.batchSize
lines.append('batch_size = ' + str(batch_size))
in_channels = args.inChannels
out_channels = args.outChannels
lstm_hidden_size = args.lstmHiddenSize #classify into three types
lines.append('lstm_hidden_size = ' + str(lstm_hidden_size))
lstm_layers = args.lstmLayers
lines.append('lstm_layers = ' + str(lstm_layers))
lstm_p = args.lstmDropout
lines.append('lstm_p = ' + str(lstm_p))
convFeatures = args.convFeatures #features of x, the flattened cnn features
lines.append('convFeatures = ' + str(convFeatures))
imuFeatures = args.IMUFeatures 
lines.append('imuFeatures = ' + str(imuFeatures))
cnn_layers = args.cnnLayers
lines.append('cnn_layers = ' + str(cnn_layers))
kernel_size = args.kernelSize
lines.append('kernel_size = ' + str(kernel_size))
conv_p = args.convDropout
lines.append('conv_p = ' + str(conv_p))
img_height = args.imgHeight
img_width = args.imgWidth
l2_reg = args.l2Reg
lines.append('l2_reg = ' + str(l2_reg))
num_workers = args.numWorkers
classes = args.classes
lines.append('classes = ' + str(classes))
seq_cap = args.seqCap
lines.append('seq_cap = ' + str(seq_cap))
prefix = args.prefix
debouncing_lim = args.debouncingLimit
lines.append('debouncing_lim = ' + str(debouncing_lim))
stack_size = args.stackSize
lines.append('stack_size = ' + str(stack_size))
sliding_window = args.slidingWindow
lines.append('sliding_window = ' + str(sliding_window))
img_folder_name = args.imgName
lines.append('img_folder_name:' + str(img_folder_name))
save_model = args.saveModel
lines.append('save_model = ' + str(save_model))
shoulder_states = args.shoulderStates
lines.append('shoulder_states = ' + str(shoulder_states))
breathing_states = args.breathingStates
lines.append('breathing_states = ' + str(breathing_states))
position_states = args.positionStates
lines.append('position_states = ' + str(position_states))
motion_states = args.motionStates
lines.append('motion_states = ' + str(motion_states))

if save_model:
  # model saving directories
  saves_folder = './model_saves'
  # create model saving dir if need
  if not os.path.exists(saves_folder):
    os.makedirs(saves_folder)
  # make folder for this run
  if not os.path.exists(saves_folder + '/OF_lstm_' + dt_string):
    os.makedirs(saves_folder + '/OF_lstm_' + dt_string)

# tensorboard setup
logdir = 'runs/'+ 'OF_lstm_' + dt_string+ 'SLID' + str(learning_rate)[2:] + '_cf' + str(convFeatures) + '_lh' + str(lstm_hidden_size) + '_ll' + str(lstm_layers) + '_sc' + str(seq_cap) + '_L2r' + str(l2_reg) + '_bs' + str(batch_size) + prefix + '_cp' + str(conv_p) + '_lp' + str(lstm_p) + args.label
writer = SummaryWriter(log_dir=logdir)
'''
end region - Args parsing and variables setup
'''

'''
region - Setup GPU
'''
gpus = args.gpus
available_gpus = [i for i in range(torch.cuda.device_count())] #'cuda:'+str(i) 
gpus_to_use = []
for g in gpus:
    if g in available_gpus:
        gpus_to_use.append(g)
primary_gpy = 'cuda:' + str(gpus_to_use[0])
device = torch.device(primary_gpy if torch.cuda.is_available() else 'cpu')
'''
end region - Setup GPU
'''


'''
region - Setup datasets
'''

#list of all sessions used in training, validation, and testing
csv_path = '~/ommdb/optical_flow/session_classes_clean.csv' 

sessions_obj = sessionsClass(shoulder_states, breathing_states, position_states, csv_path, val_pids = val_pids, test_pids = test_pids)
train_pids = sessions_obj.get_train_pids()
val_pids = sessions_obj.get_val_pids()
test_pids = sessions_obj.get_test_pids()

#get norm values
data_root = '/data/ommdb/10Hz'
norm_values = sessions_obj.get_optical_flow_norm_values_running(data_root, img_folder_name)
lines.append('OF normalization values:' + str(norm_values))

imu_norm_values = sessions_obj.get_imu_mean_std(data_root)
lines.append('imu_norm_values' + str(imu_norm_values))

body_norm_values = sessions_obj.get_train_motor_norm_values(data_root)
lines.append('body motor standardization values:' + str(body_norm_values))

body_min_max_values = sessions_obj.get_train_motor_min_max(data_root)
lines.append('body motor normalization values (min/max):' + str(body_min_max_values))

if motion_states == 'body_binary':
  body_motor_binary = True
else:
  body_motor_binary = False

#Create train, validation, and test datasets
dataset = OF_Dataset_sequenced_sliding(train_pids, seq_cap, stack_size, classes, img_folder_name, sessions_obj, body_min_max_values = body_min_max_values, norm_values = norm_values, sliding_window = sliding_window, body_motor_binary = body_motor_binary, imu_norm_values = imu_norm_values)
sample_input = dataset.__getitem__(0)[0]

dataset_val = OF_Dataset_sequenced_sliding(val_pids, seq_cap, stack_size, classes, img_folder_name, sessions_obj, body_min_max_values = body_min_max_values, norm_values = norm_values, sliding_window = sliding_window, body_motor_binary = body_motor_binary, imu_norm_values = imu_norm_values)

dataset_test = OF_Dataset_sequenced_sliding(test_pids, seq_cap, stack_size, classes, img_folder_name, sessions_obj, body_min_max_values = body_min_max_values, norm_values = norm_values, sliding_window = sliding_window, body_motor_binary = body_motor_binary, imu_norm_values = imu_norm_values)

#Padding helper function for custom dataset
#https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
def pad_collate(batch):
  (xx, yy, length) = zip(*batch)

  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=-1000)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1000)
  #print('shape padded input', xx_pad.shape)
  #print('shape padded output', yy_pad.shape)

  #pull frame from padding to test
  if img_show == True:
    input('text frame from padding')
    img = xx_pad[0][0]
    plt.imshow(img)
    plt.show()
  
  #preprae channels for cnn
  if in_channels == 1:
    xx_pad = xx_pad[:,:,None,:,:]

  else:
    xx_pad = xx_pad.permute(0,1,4,2,3)

  #print('check x_pad channel dims', xx_pad.shape)
  #input('go')

  return xx_pad, yy_pad, torch.Tensor(x_lens), torch.Tensor(y_lens)
'''
end region - Setup datasets
'''

'''
region - Initialize models
'''
#dummy data creation
sample_data, sample_labels, sample_data_lens, imu_linear, imu_gyro, imu_euler, body_motor = dataset.__getitem__(6)
dummy_input = torch.randn(sample_data.shape)
dummy_input = torch.unsqueeze(torch.Tensor(sample_data), dim=0) #add fake batch
dummy_lens = torch.tensor([sample_data_lens])
dummy_body_motor = torch.unsqueeze(torch.ones(body_motor.shape), dim = 0)
dummy_imu_linear = torch.unsqueeze(torch.ones(imu_linear.shape), dim = 0)
dummy_imu_gyro = torch.unsqueeze(torch.ones(imu_gyro.shape), dim = 0)


imu_features = imuFeatures
cnn = CNN_2(in_channels, out_channels, kernel_size, cnn_layers, convFeatures, conv_p)
if motion_states == 'both':
  lines.append('including both imu adnd motor values info in model')
  lines.append('imu features:' + str(imu_features))
  imu = IMUNet(6, imu_features)
  lstm = LSTM(lstm_hidden_size, lstm_layers, classes, convFeatures + 1 + imu_features, lstm_p) # +1 added when using body motion!!
  model = FullModel_both(cnn, lstm, imu, batch_size, img_height, img_width)
elif motion_states == 'body_motor' or motion_states == 'body_binary':
  lines.append('including motor motor info in model')
  lstm = LSTM(lstm_hidden_size, lstm_layers, classes, convFeatures + 1, lstm_p) # +1 added when using body motion!!
  model = FullModel_body_motion(cnn, lstm, batch_size, img_height, img_width)
elif motion_states == 'imu':
  lines.append('just using IMU in model')
  lines.append('imu features:' + str(imu_features))
  imu = IMUNet(6, imu_features)
  lstm = LSTM(lstm_hidden_size, lstm_layers, classes, convFeatures+imu_features, lstm_p)
  model = FullModel_imu(cnn, lstm, imu, batch_size, img_height, img_width)
elif motion_states == 'none':
  lines.append('no motion data used in model!')
  lstm = LSTM(lstm_hidden_size, lstm_layers, classes, convFeatures, lstm_p)
  model = FullModel_no_motion(cnn, lstm, batch_size, img_height, img_width)

model.forward(dummy_input, dummy_lens, dummy_body_motor, dummy_imu_linear, dummy_imu_gyro)

parallel_model = nn.DataParallel(model, device_ids = gpus_to_use) 
'''
end region - INITIALIZE MODELS
'''

'''
region - Setup Data Loaders
'''
#find train, val, and test number of videos and batches
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers) 
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers = num_workers)
lines.append('num training seqs ' + str(len(dataset)))
lines.append('num validation seq ' + str(len(dataset_val)))
lines.append('num test seqs ' + str(len(dataset_test)))
writer.add_text("Model Info", '\n'.join(lines), 0)
torch.manual_seed(42)
random.seed(42)
'''
end region - Setup Data Splitting
'''

'''
region - Training Function
'''
def train(model, lr = learning_rate, num_epochs = num_epochs, batch_size = batch_size):

  #setup loss and regularization
  if l2_reg > 0:
      print("using L2")
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = l2_reg)
  else:
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  CELoss = nn.CrossEntropyLoss(ignore_index = -1000) 

  # initialize metrics
  best_val_loss = 1000
  debouncing_count = 0
  early_stopped = False

  # iterate over epochs
  for ep in range(num_epochs):
      #check for early stopping
      if early_stopped:
        break
      #ensure gradients
      model.train()
      true_losses = []
      # iterate over batches
      lines.append('~~~~~~~~~~~~~~~ EPOCH ' + str(ep) + ' ~~~~~~~~~~~~~~')
      for batch_indx, batch in enumerate(tqdm(train_loader)):
          #unpack batch
          data, labels, data_lens, imu_linear, imu_gyro, imu_euler, body_motor = batch

          # data input testing 
          if img_show == True:
            if batch_indx == 0 and ep % 10 == 0:
              plt.figure()
              f, axarr = plt.subplots(2,3) 
              imgs = torch.permute(data[0],(0, 2, 3, 1))
              axarr[0][0].imshow(imgs[0])
              axarr[0][1].imshow(imgs[1])
              axarr[0][2].imshow(imgs[2])
              axarr[1][0].imshow(imgs[-4])
              axarr[1][1].imshow(imgs[-3])
              axarr[1][2].imshow(imgs[-2])
              plt.show()
          
          data, labels, data_lens = data.float().to(device), labels.to(device).long(), data_lens.to(device)
          imu_linear, imu_gyro, imu_euler, body_motor = imu_linear.to(device).float(), imu_gyro.to(device).float(), imu_euler.to(device).float(), body_motor.to(device).float()
          rand_h0 = torch.rand(lstm_layers, len(data), lstm_hidden_size).to(device)
          rand_c0 = torch.rand(lstm_layers, len(data), lstm_hidden_size).to(device)
          output_padded, output_lengths, hidden_states, cell_states = model(data, data_lens, body_motor, imu_linear, imu_gyro, rand_h0, rand_c0)
        
          #reshape for CE loss input: (batch, #classes, ...), target: (batch...), set to type Long
          loss = CELoss(output_padded.permute(0, 2, 1), labels) #these are both padded, but CE loss set to ignore -1000 padding value!!
          true_losses.append(loss.item())

          # backpropagate the loss
          loss.backward()

          # update parameters
          optimizer.step()

          # reset gradients
          optimizer.zero_grad()

      #Epoch-wide
      #get training specs
      train_loss, train_accuracy, train_confusion, train_f1 = eval(model, train_loader)
      lines.append('epoch train loss ' + str(train_loss))
      lines.append('epoch train_accuracy ' + str(train_accuracy))
      lines.append('epoch train_f1 ' + str(train_f1))
      lines.append('epoch train_confusion ' + str(train_confusion))

      # get val stats
      val_loss, val_accuracy, val_confusion, val_f1 = eval(model, val_loader)

      # add stats to tensorboard
      writer.add_scalar("Loss/train", train_loss, ep)
      writer.add_scalar("Accuracy/train", train_accuracy, ep)
      writer.add_scalar("f1/train", train_f1, ep)
      writer.add_text("Train Confusion", str(train_confusion), ep)
      writer.add_scalar("Loss/val", val_loss, ep)
      writer.add_scalar("Accuracy/val", val_accuracy, ep)
      writer.add_scalar("f1/val", val_f1, ep)
      writer.add_text("Val Confusion", str(val_confusion), ep)

      #test for early stopping
      lines.append('loss diff' + str(val_loss - best_val_loss))
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        debouncing_count = 0 #reset the debouncing count
        test_loss, test_acc, test_confusion, test_f1 = eval(model, test_loader)
        writer.add_scalar("Loss/test", test_loss, ep)
        writer.add_scalar("Accuracy/test", test_acc, ep)
        writer.add_scalar("f1/test", test_f1, ep)
        writer.add_text("Test Confusion", str(test_confusion), ep)
        lines.append('test_loss ' + str(test_loss))
        lines.append('test_acc ' + str(test_acc))
        lines.append('test_f1 ' + str(test_f1))
        lines.append('test_confusion ' + str(test_confusion))
        if save_model:
          torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, saves_folder + '/OF_lstm_' + dt_string + '/ep' + str(ep) + '.pt')      
      else:
        debouncing_count += 1 #increment debounding
      if debouncing_count >= debouncing_lim:
          early_stopped = True

      lines.append('val loss ' + str(val_loss) + 'at debouncing count = ' + str(debouncing_count) + '<' + str(debouncing_lim))
      lines.append('val_accuracy ' + str(val_accuracy))
      lines.append('val_f1 ' + str(val_f1))
      lines.append('val_confusion ' + str(val_confusion))

  
  #After all epochs: final test! 
  test_loss, test_acc, test_confusion, test_f1 = eval(model, test_loader)
  
  #print results and add to tensorboard
  print('~~~~~~~~~ TESTING RESULTS ~~~~~~~~')
  print('average test set loss', test_loss)
  lines.append('test_loss ' + str(test_loss))
  print('test accuracy', test_acc)
  lines.append('test_acc ' + str(test_acc))
  print(test_confusion)
  lines.append('test_confusion ' + str(test_confusion))
  writer.add_text("Final Test Results", str(test_confusion) + '\n\n' + 'test accuracy:' + str(test_acc) + '\n\n' + 'test loss:' + str(test_loss), ep)
'''
end region - Training Function
'''

'''
region - Evaluation functions
'''  
def seq_as_labels(seq):
    out = torch.argmax(seq, dim = 1)
    return out.cpu().numpy()

def eval(model, test_loader):
    model.eval()
    test_losses = []
    running_pred = []
    running_labels = []
    CELoss = nn.CrossEntropyLoss(ignore_index = -1000)
    CELoss.to(device)
    with torch.no_grad():
        for batch_indx, batch in enumerate(tqdm(test_loader)):
            #unpack data from batch and send to gpu
            data, labels, data_len, imu_linear, imu_gyro, imu_euler, body_motor = batch
            data, labels, data_len = data.float().to(device), labels.to(device).long(), data_len.to(device)
            imu_linear, imu_gyro, imu_euler, body_motor = imu_linear.to(device).float(), imu_gyro.to(device).float(), imu_euler.to(device).float(), body_motor.to(device).float()
            
            #run model to get predictions
            rand_h0 = torch.rand(lstm_layers, len(data), lstm_hidden_size).to(device)
            rand_c0 = torch.rand(lstm_layers, len(data), lstm_hidden_size).to(device)
            output_padded, output_lengths, hidden_states, cell_states = model(data, data_len, body_motor, imu_linear, imu_gyro, rand_h0, rand_c0)

            #for each video/label pair...gather the predictions vs. ground truth and their loss
            batch_loss = 0
            for i, padded_seq in enumerate(output_padded):
                # get the index for original sequence length
                pad_i = output_lengths[i]
                pad_i = pad_i.item()

                # unpad outputs and labels
                seq_output = padded_seq[:pad_i] #[num frames, 4]
                seq_labels = labels[i][:pad_i]  #[num frames, 1]

                #calculate loss and store things
                #reshape for CE loss -  input: (batch, #classes, ...), target: (batch...)
                seq_output_for_loss = seq_output.permute(1,0)
                loss = CELoss(torch.unsqueeze(seq_output_for_loss, dim = 0), torch.unsqueeze(seq_labels, dim = 0)) #unsqueeze for batch size = 1
                test_losses.append(loss.item())

                #log the predictions and ground truth for accuracy calcs
                running_pred.append(seq_as_labels(seq_output.cpu()))
                running_labels.append(seq_labels.cpu()) 
    
    #concat all the predictions and labels
    running_pred = np.concatenate(running_pred, axis=None)
    running_labels = np.concatenate(running_labels, axis=None)

    #calc accuracy
    avg_loss = (sum(test_losses)/len(test_losses))
    test_acc = accuracy_score(running_labels, running_pred) #y true then y pred
    f1 = f1_score(running_labels, running_pred, average='macro') #labels come first!

    #confusion matrix
    if classes == 4:
      confusion = confusion_matrix(running_pred, running_labels, labels = [0,1,2,3])
    elif classes == 3:
      confusion = confusion_matrix(running_pred, running_labels, labels = [0,1,2])
    elif classes == 2:
      confusion = confusion_matrix(running_pred, running_labels, labels = [0,1])

    return avg_loss, test_acc, confusion, f1
'''
end region - Evaluation functions
'''


'''
region - training commands and email
'''
train(parallel_model)
writer.flush()
writer.close()

def email(to, subject, body):

  gmail_user = 'ommierobot@gmail.com'
  gmail_password = 'zkqupedjrycgevmy'

  sent_from = gmail_user

  email_text = """\
  From: %s
  To: %s
  Subject: %s

  %s
  """ % (sent_from, ", ".join(to), subject, body)

  try:
      smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
      smtp_server.ehlo()
      smtp_server.login(gmail_user, gmail_password)
      smtp_server.sendmail(sent_from, to, email_text)
      smtp_server.close()
      print ("Email sent successfully!")
  except Exception as ex:
      print ("Something went wrong….",ex)


email(['kayla.matheus@yale.edu', 'ellie.mamantov@yale.edu'], 'Model complete', '\n'.join(lines))
'''
end region - training commands and email
'''
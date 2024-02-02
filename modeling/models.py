import torchvision
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
Simple CNN with two convolutional layers, a maxpool layer, a dropout layer, and a fully connected layer
Used to process the optical flow input prior to the LSTM
'''
class CNN_simple(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, num_cnn_layers, convFeatures, conv_p):
    super(CNN_simple, self).__init__()

    # define conv layer
    self.conv1 = nn.Conv2d(in_channels, 16, kernel_size)
    self.conv2 = nn.Conv2d(16, 32, kernel_size)
    self.maxpool = nn.MaxPool2d(2, stride=2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.LazyLinear(convFeatures*3)
    self.fc2 = nn.LazyLinear(convFeatures)
    self.relu = nn.ReLU()
    self.num_layers = num_cnn_layers
    self.dropout = nn.Dropout(conv_p)

  # define forward function
  def forward(self, x):
    x = self.conv1(x) #1 >> 16
    x = self.maxpool(x)
    x = self.conv2(x) # 16 >> 32
    x = self.dropout(x)
    x = self.flatten(x)    
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    return x 


'''
Similar to the CNN_simple, but with Batch Normalization and lazy linear fully connected layers
Used to process the optical flow input prior to the LSTM
'''
class CNN_2(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, num_cnn_layers, convFeatures, conv_p):
    super(CNN_2, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, 16, kernel_size)
    self.conv2 = nn.Conv2d(16, 32, kernel_size)
    self.batchnorm1 = nn.BatchNorm2d(16)
    self.batchnorm2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 16, kernel_size)
    self.maxpool = nn.MaxPool2d(2, stride=2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.LazyLinear(convFeatures*3)
    self.fc2 = nn.LazyLinear(convFeatures)
    self.relu = nn.ReLU()
    self.num_layers = num_cnn_layers
    self.dropout = nn.Dropout(conv_p)

  # define forward function
  def forward(self, x):
    x = self.relu(self.conv1(x)) #1 >> 16
    x = self.maxpool(x)
    x = self.relu(self.conv2(x)) # 16 >> 32
    x = self.flatten(x)
    x = self.dropout(x) #Dropouts are usually advised not to use after the convolution layers, they are mostly used after the dense layers of the network.
    x = self.relu(self.fc2(x))

    return x    

''' Simple MLP to expand the dimensionality of the IMU inputs'''
class IMUNet(nn.Module):
  def __init__(self, expansion_size, out_features):
    super(IMUNet, self).__init__()
    #input size will be 6 features
    self.fc1 = nn.LazyLinear(expansion_size) #expand from 6 input features
    self.compress = nn.LazyLinear(out_features) #should be same as conv features (50)
    self.relu = nn.ReLU()

  def forward(self, imu_linear, imu_gyro):
    lin = self.relu(self.fc1(imu_linear))
    gyro = self.relu(self.fc1(imu_gyro))
    concat = torch.cat((lin, gyro), dim = 2)
    out = self.relu(self.compress(concat))
    return out

'''Core LSTM used in all models'''
class LSTM(nn.Module):
  def __init__(self, hidden_size, num_layers, classes, convFeatures, lstm_p):  #add dropout later
    super(LSTM, self).__init__()

    # input parameters 
    self.hidden_size = hidden_size # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.num_layers = num_layers # Number of layers in the lstm
    self.classes = classes # Number of output neurons 
    self.LSTM = nn.LSTM(convFeatures, self.hidden_size, self.num_layers, dropout = lstm_p, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, self.classes)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

  def forward(self, x, x_lens, hidden_state = None, hidden_cell = None):
    
    #pack the padded sequence data -- turns [batch size, seq len, H, W] into the variable seq leng data
    conv_seqs_packed = pack_padded_sequence(x, x_lens.to('cpu'), batch_first=True, enforce_sorted=False)
    #set random inits for hidden and cell states
    batch_sz = x.shape[0]

    if hidden_state == None and hidden_cell == None:
      #if there is no specified starting hidden states, then randomize
      outputs, (hidden_state, hidden_cell) = self.LSTM(conv_seqs_packed)
    else:
      #if there is specified hidden state/cell to start with, use it ()
      outputs, (hidden_state, hidden_cell) = self.LSTM(conv_seqs_packed, (hidden_state, hidden_cell))
    #pad the sequence data -- turns variable len seqs into [batch size, seq len, H, W]
    output_padded, output_lengths = pad_packed_sequence(outputs, batch_first=True)
    
    #run through linear
    out = self.fc(output_padded)
    out = self.relu(out)
    return out, output_lengths, hidden_state, hidden_cell
  

'''The full model, with neither the IMU or body motor inputs included'''
class FullModel_no_motion(nn.Module):
  def __init__(self, cnn, lstm, batch_size, height, width):
    super(FullModel_no_motion, self).__init__()

    self.cnn = cnn
    self.lstm = lstm
    self.batch_size = batch_size
    self.height = height
    self.width = width

  def forward(self, x, x_lens, body_motor, imu_linear, imu_gyro, hidden_state = None, hidden_cell = None): #x = sequence?
    # get original batch and seq len sizes
    seq_sz = x.shape[1]
    batch_sz = x.shape[0]
    chan = x.shape[2]
    h = x.shape[3]
    w = x.shape[4]
    
    x_for_cnn = torch.flatten(x, start_dim = 0, end_dim = 1)
    cnn_out = self.cnn(x_for_cnn)
    cnn_out = cnn_out.view(batch_sz, seq_sz, -1)
   
    outputs, output_lengths, hidden_state, cell_state = self.lstm(cnn_out, x_lens, hidden_state, hidden_cell) 
    return outputs, output_lengths, hidden_state, cell_state

'''The full model, with only body motor inputs included'''
class FullModel_body_motion(nn.Module):
  def __init__(self, cnn, lstm, batch_size, height, width):
    super(FullModel_body_motion, self).__init__()

    self.cnn = cnn
    self.lstm = lstm
    self.batch_size = batch_size
    self.height = height
    self.width = width

  def forward(self, x, x_lens, body_motor, imu_linear, imu_gyro, hidden_state = None, hidden_cell = None): #x = sequence?
    # get original batch and seq len sizes
    seq_sz = x.shape[1]
    batch_sz = x.shape[0]
    chan = x.shape[2]
    h = x.shape[3]
    w = x.shape[4]

    x_for_cnn = torch.flatten(x, start_dim = 0, end_dim = 1)
   
    cnn_out = self.cnn(x_for_cnn)
    cnn_out = cnn_out.view(batch_sz, seq_sz, -1)
    body_motor_for_cat = torch.unsqueeze(body_motor, dim = 2)
    motor_added = torch.cat((cnn_out, body_motor_for_cat), dim = 2) #batch size, stack size, conv features
  
    #run through lstm (input shape: [batch_size, seq_size (padded), convFeatures])
    outputs, output_lengths, hidden_state, cell_state = self.lstm(motor_added, x_lens, hidden_state, hidden_cell) 
    return outputs, output_lengths, hidden_state, cell_state

'''The full model, with only IMU inputs included'''
class FullModel_imu(nn.Module):
  def __init__(self, cnn, lstm, imu, batch_size, height, width):
    super(FullModel_imu, self).__init__()

    self.cnn = cnn
    self.lstm = lstm
    self.imu = imu
    self.batch_size = batch_size
    self.height = height
    self.width = width

  def forward(self, x, x_lens, body_motor, imu_linear, imu_gyro, hidden_state = None, hidden_cell = None): #x = sequence?
    # get original batch and seq len sizes
    seq_sz = x.shape[1]
    batch_sz = x.shape[0]
    chan = x.shape[2]
    h = x.shape[3]
    w = x.shape[4]
    
    x_for_cnn = torch.flatten(x, start_dim = 0, end_dim = 1)
    cnn_out = self.cnn(x_for_cnn)
    cnn_out = cnn_out.view(batch_sz, seq_sz, -1)

    #add in the motion bits
    imu_out = self.imu(imu_linear,imu_gyro) #size would be [batch size, seq size, convFeatures]
    imu_added = torch.cat((cnn_out, imu_out), dim = 2) #batch size, stack size, conv features
    #run through lstm (input shape: [batch_size, seq_size (padded), convFeatures])
    outputs, output_lengths, hidden_state, cell_state = self.lstm(imu_added, x_lens, hidden_state, hidden_cell) 
    return outputs, output_lengths, hidden_state, cell_state

'''The full model, with BOTH IMU and body motor inputs included'''
class FullModel_both(nn.Module):
  def __init__(self, cnn, lstm, imu, batch_size, height, width):
    super(FullModel_both, self).__init__()

    self.cnn = cnn
    self.lstm = lstm
    self.imu = imu
    self.batch_size = batch_size
    self.height = height
    self.width = width

  def forward(self, x, x_lens, body_motor, imu_linear, imu_gyro, hidden_state = None, hidden_cell = None): #x = sequence?
    # get original batch and seq len sizes
    seq_sz = x.shape[1]
    batch_sz = x.shape[0]
    chan = x.shape[2]
    h = x.shape[3]
    w = x.shape[4]
    x_for_cnn = torch.flatten(x, start_dim = 0, end_dim = 1)    
    cnn_out = self.cnn(x_for_cnn)

    #put tback into sequences
    cnn_out = cnn_out.view(batch_sz, seq_sz, -1)

    #add in the motion bits
    imu_out = self.imu(imu_linear,imu_gyro) #size would be [batch size, seq size, convFeatures]
    imu_added = torch.cat((cnn_out, imu_out), dim = 2) #batch size, stack size, conv features

    body_motor_for_cat = torch.unsqueeze(body_motor, dim = 2)
    both_added = torch.cat((imu_added, body_motor_for_cat), dim = 2) #batch size, stack size, conv features
  
    #run through lstm (input shape: [batch_size, seq_size (padded), convFeatures])
    outputs, output_lengths, hidden_state, cell_state = self.lstm(both_added, x_lens, hidden_state, hidden_cell) 
  
    return outputs, output_lengths, hidden_state, cell_state


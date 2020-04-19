# Import Python libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

from model import CNN_MFembedded_BCM
from utilities import predict
##########################################################################################
# Set up GPU for running
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##########################################################################################
# Load data
train_dataX = np.load('./data/DILI_train_fp_2048.npy') #1597
train_dataY = np.load('./data/DILI_train_Y_2048.npy') # 1597
test_dataX  = np.load('./data/DILI_test_fp_2048.npy') #322
test_dataY  = np.load('./data/DILI_test_Y_2048.npy') # 322

##########################################################################################
# Set up training parameters
max_len        = 220 # (m)
bit_size       = 2048 # (n)
embedding_size = 220 #(k)
window_height  = 5
dropout_rate   = 0.5
num_filter     = 4096
batch_size     = 128

##########################################################################################
# Create index vector of compound and padding zero for TRAINING DATA
train_data_x = []
train_data_y = []
for i in range(len(train_dataX)):
    fp = [0] * max_len
    n_ones = 0
    for j in range(bit_size):
        if train_dataX[i][j] == 1:
            fp[n_ones] = j+1
            n_ones += 1
    train_data_x.append(fp)
    train_data_y.append([train_dataY[i]])
    
train_data_x = np.array(train_data_x, dtype=np.int32)
train_data_y = np.array(train_data_y, dtype=np.float32)

##########################################################################################
# Creating index vector of compound and padding zero for TEST DATA
test_data_x = []
test_data_y = []
for i in range(len(test_dataX)):
    fp = [0] * max_len
    n_ones = 0
    for j in range(bit_size):
        if test_dataX[i][j] == 1:
            fp[n_ones] = j+1
            n_ones += 1
    test_data_x.append(fp)
    test_data_y.append([test_dataY[i]])
    
test_data_x = np.array(test_data_x, dtype=np.int32)
test_data_y = np.array(test_data_y, dtype=np.float32)

##########################################################################################
# Split data into training and validation set
seed = 10
x_train, x_val, y_train, y_val = train_test_split(train_data_x, train_data_y, test_size = 0.1, random_state = seed, shuffle = True)
x_test, y_test = test_data_x, test_data_y

# Create feature and targets tensor for training set, validation set, test set
featuresTrain = torch.from_numpy(x_train).type(torch.LongTensor) # 1277
targetsTrain  = torch.from_numpy(y_train)

featuresVal   = torch.from_numpy(x_val).type(torch.LongTensor) # 320
targetsVal    = torch.from_numpy(y_val)

featuresTest  = torch.from_numpy(x_test).type(torch.LongTensor) # 322
targetsTest   = torch.from_numpy(y_test)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
val   = torch.utils.data.TensorDataset(featuresVal,targetsVal)
test  = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# Data loader
training_loader     = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
validation_loader   = torch.utils.data.DataLoader(val,   batch_size = batch_size, shuffle = False)
test_loader         = torch.utils.data.DataLoader(test,  batch_size = batch_size, shuffle = False)

##########################################################################################
# Create Model
model = CNN_MFembedded_BCM(bit_size, embedding_size, max_len, window_height, num_filter, dropout_rate).to(device)
# Cross Entropy Loss 
criteron = nn.BCELoss()
# Adam Optimizer
learning_rate = 0.0025
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##########################################################################################

##########################################################################################                 
###                  DEFINE TRAINING, VALIDATION, AND TEST FUNCTION                    ###           
##########################################################################################

# Training Function
def train(epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, labels) in enumerate(training_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data).view_as(labels)
        #------------------- 
        optimizer.zero_grad()
        #------------------- 
        loss = criteron(outputs, labels)  
        loss.backward()
        train_loss += loss.item()*len(data) #(loss.item is the average loss of training batch)
        optimizer.step() 
        #-------------------
        predicted = predict(outputs, len(data)).to(device)
        train_correct += (predicted == labels).sum()
        #-------------------
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(training_loader.dataset),
                100. * batch_idx / len(training_loader),
                loss.item()))
    #------------------- 
    print('====> Epoch: {} Average Train Loss: {:.4f}'.format(epoch, train_loss / len(training_loader.dataset)))
    print('====> Epoch: {} Average Train Acuracy: {:.4f}'.format(epoch, train_correct.float() / len(training_loader.dataset)))
    #------------------- 
    # Save the trained models
    #PATH = './checkpoint/seed{}/e{}_k{}_m{}_h{}_lr{}_seed{}.pth'.format(seed, epoch, embedding_size, max_len, window_height, learning_rate, seed)
    #PATH = './checkpoint/seed{}/add_e{}_k{}_m{}_h{}_lr{}_seed{}.pth'.format(seed, epoch, embedding_size, max_len, window_height, learning_rate, seed)
    #torch.save(model.state_dict(), PATH)         
    #-------------------
    train_loss = (train_loss / len(training_loader.dataset) )
    train_acc  = (train_correct.float()/ len(training_loader.dataset) )
    train_pred = outputs
    #-------------------
    return train_loss, train_acc

##########################################################################################
# Validation Function
def validate(epoch):
    model.eval()
    validation_loss = 0
    validation_correct = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_loader): 
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data).view_as(labels)
            loss = criteron(outputs, labels)
            validation_loss += loss.item()*len(data)
            predicted = predict(outputs, len(data)).to(device)
            validation_correct += (predicted == labels).sum()
    #------------------- 
    validation_loss = (validation_loss / len(validation_loader.dataset) )
    validation_acc = (validation_correct.float()/ len(validation_loader.dataset) )
    #------------------- 
    print('====> Average Validation Loss: {:.4f}'.format(validation_loss))
    print('====> Epoch: {} Average Validation Acuracy: {:.4f}'.format(epoch, validation_acc))
    #-------------------
    return validation_loss, validation_acc
    
##########################################################################################
# Test Function
def test(epoch):
    model.eval()
    test_loss = 0
    test_correct = 0
    pred_prob = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader): 
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data).view_as(labels)
            loss = criteron(outputs, labels)
            test_loss += loss.item()*len(data)
            pred_prob.append(outputs)
            predicted = predict(outputs, len(data)).to(device)
            test_correct += (predicted == labels).sum()
    #------------------- 
    test_loss = (test_loss / len(test_loader.dataset) )
    test_acc  = (test_correct.float()/ len(test_loader.dataset) )
    #------------------- 
    print('====> Average Test Loss: {:.4f}'.format(test_loss))
    print('====> Epoch: {} Average Test Acuracy: {:.4f}'.format(epoch, test_acc))
    #-------------------
    return test_loss, test_acc, pred_prob, predicted

##########################################################################################                 
###                                  TRAINING MODEL                                    ###           
##########################################################################################
training_loss_list, validation_loss_list, test_loss_list = [], [], []
training_acc_list, validation_acc_list, test_acc_list = [], [], []
x_axis = []
test_prob_pred = []

#################################################################################
for epoch in range(1, 21):
    #torch.manual_seed(1)
    x_axis.append(str(epoch))
    #start = time.time()
    train_results = train(epoch)
    #end = time.time()
    validation_results = validate(epoch)
    #start = time.time()
    test_results = test(epoch)
    #end = time.time()
    #------------------- 
    training_loss_list.append(train_results[0])
    validation_loss_list.append(validation_results[0])
    test_loss_list.append(test_results[0])
    #-------------------
    training_acc_list.append(train_results[1])
    validation_acc_list.append(validation_results[1])
    test_acc_list.append(test_results[1])
    #-------------------
    test_prob_pred.append(test_results[2])

#################################################################################

# Import Python libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

##########################################################################################
class CNN_MFembedded_BCM(nn.Module):
    torch.manual_seed(1)
    def __init__(self, bit_size = 1024, embedding_size = 100, max_length = 100, window_height = 5, num_filter = 1024, dropout_rate=0.5):
        super(CNN_MFembedded_BCM, self).__init__()
        #Network Architecture
        self.embeddings     = nn.Embedding(bit_size + 1, embedding_size, padding_idx = 0) #(100, 100)
        self.bnorm_emb      = nn.BatchNorm2d(num_features = 1)   
        #####################
        self.conv1          = nn.Conv2d(in_channels = 1, out_channels = num_filter, kernel_size = (window_height, embedding_size)) # (1024, 96, 1)
        self.bnorm_conv1    = nn.BatchNorm2d(num_features = num_filter)
        self.activate_conv1 = nn.LeakyReLU()
        self.pool_conv1     = nn.MaxPool2d(kernel_size = (max_length - window_height + 1, 1))
        #self.drop_conv1     = nn.Dropout(p=dropout_rate)
        #####################
        self.linear1        = nn.Linear(num_filter, 512)
        self.bnorm_l1       = nn.BatchNorm1d(num_features = 1)
        self.activate_l1    = nn.LeakyReLU()
        self.drop1          = nn.Dropout(p=dropout_rate)
        #####################
        self.linear2        = nn.Linear(512, 256)
        self.bnorm_l2       = nn.BatchNorm1d(num_features = 1)
        self.activate_l2    = nn.LeakyReLU()
        self.drop2          = nn.Dropout(p=dropout_rate)
        #####################
        self.linear3        = nn.Linear(256, 1)
        self.activate_l3    = nn.Sigmoid()
        #####################
        #Variables
        self.embedded_size  = embedding_size
        self.max_length     = max_length
        self.pooled_size    = num_filter
        
    def forward(self, x):
        torch.manual_seed(1)
        embeds              = self.embeddings(x).view(-1, 1, self.max_length, self.embedded_size)
        embeds              = self.bnorm_emb(embeds)
        #####################
        output1             = self.conv1(embeds)
        output1             = self.bnorm_conv1(output1)
        output1             = self.activate_conv1(output1)
        output1             = self.pool_conv1(output1).view(-1, 1, self.pooled_size) 
        #output1             = self.drop_conv1(output1)
        #####################
        output2             = self.linear1(output1) 
        output2             = self.bnorm_l1(output2) 
        output2             = self.activate_l1(output2) 
        output2             = self.drop1(output2) 
        #####################
        output3             = self.linear2(output2)
        output3             = self.bnorm_l2(output3) 
        output3             = self.activate_l2(output3)
        output3             = self.drop2(output3)            
        #####################
        output4             = self.linear3(output3)
        output4             = self.activate_l3(output4)
        return output4
##########################################################################################

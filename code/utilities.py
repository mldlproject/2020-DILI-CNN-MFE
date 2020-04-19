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

##########################################################################################
def predict(X, batch_size = 128):
    class_list = []
    for i in X:
        if i >= 0.5:
            class_list.append(1)
        else:
            class_list.append(0)
    class_list = torch.FloatTensor(class_list).view(batch_size, 1)
    return class_list

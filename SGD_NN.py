# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 21:24:42 2019

PyTorch neural network based stochastic gradient descent to optimize a similarity matrix for peptide binding affinity

@author: dhyla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict

''' returns a 12 x 20 one hot encoding of peptide '''
AA=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
loc=['N','2','3','4','5','6','7','8','9','10','11','C']
def oneHotter(peptide):
    aminoAcids = list(peptide)  # below: initialize ordered dictionary to hold empty lists of amino acid length
    aaDict = OrderedDict()
    aaDict = {"A": [], "R": [], "N": [], "D": [], "C": [], "Q": [], "E": [], "G": [], "H": [], "I": [],
              "L": [], "K": [], "M": [], "F": [], "P": [], "S": [], "T": [], "W": [], "Y": [], "V": []}
    for index in range(len(aminoAcids)):
        for key, value in aaDict.items():
            if (aminoAcids[index] == key):
                value.append(1)
            else:
                value.append(0)
    result = torch.Tensor(list(aaDict.values()))
    return result

''' returns tuple of unique pair randomly sampled '''
usedPairs = [] 
def findPair(data):
    peps = data.sample(2)
    pep1 = peps.index[0]
    pep2 = peps.index[1]
    pair = (pep1, pep2)
    if pair not in usedPairs:
        usedPairs.append(pair)
        return pair
    else:
        return findPair(data) # RECURSION
    
''' NN class defined with 4 layers '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(400, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5,1)
    
    # takes in a matrix as an input and returns the 
    # result of the forward propogation 
    def forward(self, x):
        x = torch.reshape(x, (1, 400)) # flattens matrix 
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return (abs(x))

model = Net()
loss_fn = torch.nn.SmoothL1Loss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# file input
dirname = 'C:/Users/dhyla/OneDrive/Desktop/Research/Data/'
data = pd.read_csv(dirname + 'cleaned_set1.csv')
data.set_index('AA_seq',inplace=True)

model.train()

# train, test split
train_data, test_data = train_test_split(data, test_size=0.2)

# Calculates and optimizes a similarity matrix using NNs
num_epochs = 5
for epoch in range(num_epochs):
    for i in range(len(data)):
        pepTuple = findPair(train_data)
        pep1 = pepTuple[0]
        pep2 = pepTuple[1]
        y1 = np.round(train_data.binding_affinity[train_data.index == pep1].iloc[0],decimals=4)
        y2 = np.round(train_data.binding_affinity[train_data.index == pep2].iloc[0],decimals=4)
        if y1 - y2 == 0:  # max case
            Y = torch.FloatTensor([10001])
        else:
            Y = torch.FloatTensor([np.abs(1 / (y1 - y2))])
        sim = torch.mm(oneHotter(pep1), oneHotter(pep2).t())
        # forward pass       
        y_pred = model.forward(sim)
        loss = loss_fn(y_pred, Y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch:", epoch, " Iteration:", i, " Loss:", loss.item())















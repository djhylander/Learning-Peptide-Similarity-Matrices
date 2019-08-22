# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 01:21:01 2019

@author: dhyla
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
from torch.autograd import Variable as Var
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split

''' file input '''
dirname = 'C:/Users/dhyla/OneDrive/Desktop/Research/Data/'
data = pd.read_csv(dirname + 'cleaned_set1.csv')
data.set_index('sequence',inplace=True)


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
    result = np.array(list(aaDict.values()))
    return result

'''train, test split '''
train_data, test_data = train_test_split(data, test_size=0.2)

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
        return findPair(data)

'''calculates the similarity matrix for given initial matrix'''
def calcSim(preSim):
    preSim.data = (preSim.data - preSim.min().data) / (preSim.max().data - preSim.min().data) * 2 - 1  # normalize the preSim matrix
    simMat = torch.mm(preSim, preSim.t())
    simDiag = 1 - torch.diag(simMat)
    simDiag = torch.reshape(simDiag,[20,1])
    diagMat = torch.sqrt(torch.mm(simDiag,simDiag.t()))
    simMat.data = simMat.data+diagMat.data
    simMat.data = (simMat.data - simMat.min().data) / (simMat.max().data - simMat.min().data) * 2 - 1  #normalize the similarity matrix --> to get [-1,1]
    return simMat

''' gradient descent '''
torch.cuda.set_device(0)
preSim = Var(torch.randn(20,1).cuda(), requires_grad=True)
preSim_cuda = preSim.cuda()
constant = Var(torch.randn(1,1).cuda(),requires_grad=True)
constant_cuda = constant.cuda()
exponent = Var(torch.randn(1,1).cuda(),requires_grad=True)
exponent_cuda = exponent.cuda()

optimizer = optim.Adam([preSim]+[constant]+[exponent],lr=1e-3,weight_decay=0.1)
loss_fn = nn.SmoothL1Loss()

mse_epochwise = []
epoch = 1
while epoch <= 1:
    mseAll=[]
    for i in range(len(train_data)):
        pepTuple = findPair(train_data)
        pep1 = pepTuple[0]
        pep2 = pepTuple[1]
        simMat = calcSim(preSim)
        mask = Var(torch.Tensor(np.dot(oneHotter(pep1), np.transpose(oneHotter(pep2)))).cuda(), requires_grad=False) # create mask to apply on similarity matrix
        hotmask = mask * simMat 
        pss = torch.sum(hotmask)
        y1 = np.round(train_data.meas[train_data.index == pep1].iloc[0],decimals=4)
        y2 = np.round(train_data.meas[train_data.index == pep2].iloc[0],decimals=4)
        if y1 - y2 == 0:  # max case
            Y = torch.FloatTensor([10001])
        else:
            Y = torch.FloatTensor([np.abs(1 / (y1 - y2))])
        Y1 = torch.mul(constant.cpu(),torch.pow(Y,exponent.cpu()))
        mse = loss_fn(Y1.cuda(),pss).cpu() # difference between binding_affinity is proportional to 1/total-similarity-score. so 'their' difference is the error
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        mseAll.append(np.array(mse.data))
        if np.array(mse.data) == min(mseAll): #save best model
            bestSim = calcSim(preSim)
        print("epoch:", epoch, " Iteration:", i, "  MSE:", round(mse.data.item(), 3), "  k:", round(constant.data.item(), 3), "  exp:", round(exponent.data.item(), 3))
    epoch += 1
    mse_epochwise.append(np.mean(mseAll))


''' All graphing stuff '''

#print(bestsim.data) # show bestmodel
bestsimfig = sns.heatmap(bestSim.data.cpu(), xticklabels = AA, yticklabels = AA)#visualize
fig = bestsimfig.get_figure()
fig.savefig(dirname + '/Plots/bestSim_nonlinear_allvars.png')
#plt.show()
plt.close()
np.savetxt('best_sim_nonlinear_allvars.txt',np.asarray(bestSim.data.cpu()),delimiter='\t')

newsimmat = simMat #last iteration model
np.savetxt('new_sim_nonlinear_allvars.txt',np.asarray(newsimmat.data.cpu()),delimiter='\t')
newsimfig = sns.heatmap(newsimmat.data.cpu(), xticklabels = AA, yticklabels = AA)#visualize
fignewsim = newsimfig.get_figure()
fignewsim.savefig(dirname + '/Plots/newSim_nonlinear_allvars.png')
#plt.show()
plt.close()

extra = plt.scatter(x = np.linspace(0, len(mse_epochwise),num = len(mse_epochwise)), y = mse_epochwise)
plt.xlabel(xlabel = 'iterations')
plt.ylabel(ylabel = 'Mean Squared Error')
plt.ylim(0, max(mse_epochwise))
plt.savefig(dirname + '/Plots/MSE_nonlinear_allvars.png')
#plt.show()
plt.close()



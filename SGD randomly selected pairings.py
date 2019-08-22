# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 20:29:58 2019

@author: dhyla

Calculates the most ideal similarity matrix using randomly selected unique pairings of a list of peptides until below a certain error
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
#import time

# the avergae time (1000 attempts) taken to calc all unique onehot pairings for 10 peptides: 0.055 seconds
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

# returns the Pairwise Similarity Score for two peptides
def findPSS():
    return np.multiply(np.dot(oneHot1, oneHot2), simMatrix).sum()
        
# Calculates the gradient to decrease the similarity matrix by
def findGradient():  # Error = |TrueAffinity - pi *pj * S|^2  
    gradient = -2 * (actualAffinity - findPSS()) * np.dot(oneHot1, oneHot2).sum()  # Derivative = -2 * |TrueAffinity - pi *pj * S| * (pi * pj)   
    return gradient 
    
def minMatrix():  # minimizes the similarity matrix via gradient descent 
    alpha = .01  # weight of gradient
    for i in range(100):
        gradient = findGradient()
        if (abs(gradient) > .01):  # terminate if iterations > 10000 or change in matrix < .0001
            change = alpha * gradient
            global simMatrix 
            simMatrix -= change # new similarity matrix
            alpha *= .98  # to ensure we don't skip over minimum indefinitely, don't know what is right momentum
        else:
            break
    
simMatrix = np.random.rand(20, 20)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print("Inital random sim matrix: \n\n", simMatrix)
data = pd.read_csv("peptideTest10.csv", usecols = ["AA_seq"])
pepCol = list(data["AA_seq"])  # create list using the column of peptides
numCol = []
for i in range(len(pepCol)):  # create a list with the indexes of the peptides
    numCol.append(i)
pairs = []
for n in range(1000):  # used for "infinite" loop until error is low enough 
    i = np.random.randint(0, len(pepCol))
    j = np.random.randint(0, len(pepCol))
    if (i != j):
        if ([i, j] not in pairs and [j, i] not in pairs):
            pairs.append([i, j])
    actualAffinity = 1 / abs(.6 - .5) # randomly initialzied actual binding affinity
    oneHot1 = oneHotter(pepCol[i]) # 20 x n one hot matrix for peptide 1 where n = number of unique amino acids
    oneHot2 = oneHotter(pepCol[j]).transpose()  # n x 20 one hot matrix for peptide 2
    minMatrix()
print("\nNew matrix after learning: \n\n", simMatrix)



        
        




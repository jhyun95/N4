#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:58:27 2018

@author: jhyun95
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import transform_black_and_white, ConvNet, train_model, test_model
from interactions import find_pixel_correlations
from hierarchical_clustering import compute_hierarchical_clusters

def main():
    WORKING_DIR = 'DCell_test/'
    TRUE_MODEL_NAME = 'ConvNet_E10'
    CORRELATION_MODE = 'mcc-adj'
    LABELSET = 2
    
    ''' Train the "true" image model '''
    model_path = WORKING_DIR + TRUE_MODEL_NAME
    model = ConvNet()
    if os.path.isfile(model_path):
        torch.load(model.state_dict(), model_path)
    else:
        model = train_true_model(model, epochs=10)
        torch.save(model.state_dict(), model_path)
    
    ''' From target label and compute pixel correlations '''
    corr_path = WORKING_DIR + 'pixel-' + str(LABELSET) +  \
                    '-' + CORRELATION_MODE + '.csv.gz'
    if os.path.isfile(corr_path):
        correlations = np.loadtxt(corr_path, delimiter=',')
    else:
        correlations = find_pixel_correlations(model, labelset=LABELSET, 
                        check_consistency=True, mode=CORRELATION_MODE, 
                        output_file=corr_path)
        
    ''' Construct pixel ontology from correlations '''
    distances = np.max(correlations) - np.abs(correlations)
    root, adj, assoc, unique_assoc = compute_hierarchical_clusters( 
            distances, 0.05, 10, True, False, True)
    
    ''' Initialize DCell-like network for specific image label '''
    ''' Create synthetic DCell-like knockout data '''
    ''' Train DCell-like model on synthetic knockout data'''
    ''' Evaluate DCell-like model '''
    
def train_true_model(model, input_batch=64, test_batch=1000, lr=0.01, 
                     momentum=0.5, epochs=10, print_interval=25):
    ''' Load MNIST datasets '''
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, 
                       transform=transforms.Compose(transform_black_and_white())),
        batch_size=input_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(transform_black_and_white())),
        batch_size=test_batch, shuffle=True)
    
    ''' Train/test for specified number of epochs'''
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)# weight_decay=WEIGHT_DECAY)
    for epoch in range(1, epochs + 1):
        train_model(model, train_loader, optimizer, epoch, print_interval)
        test_model(model, test_loader)
    return model

if __name__ == '__main__':
    main()
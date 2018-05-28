#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:58:27 2018

@author: jhyun95
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import transform_black_and_white, ConvNet, train_model, test_model
from interactions import find_pixel_correlations, __get_prediction__
from hierarchical_clustering import compute_hierarchical_clusters

def main():
    WORKING_DIR = 'DCell_test/'
    TRUE_MODEL_NAME = 'ConvNet_E20'; TRUE_MODEL_EPOCHS=20
    BASE_IMAGE_HEX = '0x1e000003e000007e000007e000007c000003c000000f8000000f0000003c000000600000038000001c000000e00000070001f8380007ff80001ff8000000000000000000000000000000000000000000'
    TRUE_LABEL = 5 # Image should be 5, ~14k positive interactions, ~140k negative interactions
    CORRELATION_MODE = 'mcc-adj'
    
    ''' Initialize working directory '''
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    
    ''' Train the "true" image model '''
    print('Training/Loading true image model...')
    model_path = WORKING_DIR + TRUE_MODEL_NAME
    true_model = ConvNet()
    if os.path.isfile(model_path):
        torch.load(true_model.state_dict(), model_path)
    else:
        true_model = train_true_model(true_model, epochs=TRUE_MODEL_EPOCHS)
        torch.save(true_model.state_dict(), model_path)
        
    ''' Analyze base image with respect to the true model '''
    base_label = __get_prediction__(true_model, BASE_IMAGE_HEX)
    if base_label != TRUE_LABEL:
        print('WARNING: True model incorrectly labels the base image')
        sys.exit(1)
    else:
        print('True model and label agree on base image:', base_label)
    
    ''' From target label and compute pixel correlations '''
    print('Computing pairwise pixel correlations for all images with same label...')
    corr_path = WORKING_DIR + 'pixel-' + str(base_label) +  \
                    '-' + CORRELATION_MODE + '.csv.gz'
    if os.path.isfile(corr_path):
        correlations = np.loadtxt(corr_path, delimiter=',')
    else:
        correlations = find_pixel_correlations(true_model, labelset=correlations, 
                        check_consistency=True, mode=CORRELATION_MODE, 
                        output_file=corr_path)
        
    ''' Construct pixel ontology from correlations '''
    print('Computing pixel ontology for DCell model...')
    distances = np.max(correlations) - np.abs(correlations)
    root, adj, assoc, unique_assoc = compute_hierarchical_clusters( 
            distances, 0.05, 10, True, False, True)
    
    ''' Initialize DCell-like network for specific image label '''
#    dcell_model
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
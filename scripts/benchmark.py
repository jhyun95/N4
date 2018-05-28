#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:58:27 2018

@author: jhyun95
"""

import os, sys, datetime
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from models import transform_black_and_white, ConvNet, \
    train_model, test_model, make_dcellnet_for_label
from interactions import find_pixel_correlations, __get_prediction__
from data_generator import generate_dcell_train_data, generate_dcell_eval_data

''' Global pipeline parameters '''
WORKING_DIR = '../data/DCell_test/'
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.05
MOMENTUM = 0.5

def main():
    ''' Initialize working directory and log '''
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    log_file = WORKING_DIR + 'log.txt'
    
    ''' Print to console and log file '''
    with LoggingPrinter(log_file):
        benchmark()

def benchmark():
    TRUE_MODEL_NAME = 'ConvNet_E20'
    TRUE_MODEL_EPOCHS=20
    BASE_IMAGE_HEX = '0x1e000003e000007e000007e000007c000003c000000f8000000f0000003c000000600000038000001c000000e00000070001f8380007ff80001ff8000000000000000000000000000000000000000000'
    TRUE_LABEL = 5 # Image should be 5, ~14k positive interactions, ~140k negative interactions
    CORRELATION_MODE = 'mcc-adj'
    DCELL_MODEL_NAME = 'DCell_E1'
    DCELL_MODEL_EPOCHS = 1
    TEST_DATA_SEED = 1
    CLUSTER_THRESHOLD = 0.05
    MIN_CLUSTER_SIZE = 10
    TRAINING_DOUBLE_COUNT = 100000
    VALIDATION_DOUBLE_COUNT = 20000
    EVALULATION_COUNT = 50000
    MAX_EVAL_KNOCKOUT = 10
       
    print('------ BEGIN LOG:', datetime.datetime.now(), '-----------------------------------')
    print('Base Image:', BASE_IMAGE_HEX)
    print('Base Image Label:', TRUE_LABEL)
    print('Test Data Seed:', TEST_DATA_SEED)
    
    ''' Load or train the "true" image model '''
    model_path = WORKING_DIR + TRUE_MODEL_NAME
    true_model = ConvNet()
    if os.path.isfile(model_path):
        print('Loading true image model...')
        true_model.load_state_dict(torch.load(model_path))
    else:
        print('Training true image model...')
        true_model = train_true_model(true_model, epochs=TRUE_MODEL_EPOCHS)
        torch.save(true_model.state_dict(), model_path)
    true_model.eval()
        
    ''' Analyze base image with respect to the true model '''
    base_label = __get_prediction__(true_model, BASE_IMAGE_HEX)
    if base_label != TRUE_LABEL:
        print('WARNING: True model incorrectly labels the base image')
        sys.exit(1)
    else:
        print('True model and label agree for base image:', TRUE_LABEL)
    
    ''' From target label, pre-compute pixel correlations for DCell model '''
    corr_path = WORKING_DIR + 'pixel-' + str(TRUE_LABEL) +  \
                    '-' + CORRELATION_MODE + '.csv.gz'
    if not os.path.isfile(corr_path):
        find_pixel_correlations(true_model, labelset=TRUE_LABEL, 
                                check_consistency=True, mode=CORRELATION_MODE, 
                                output_file=corr_path)
    
    ''' Initialize DCell-like network from correlations for image label '''
    print('Initializing DCell network...')
    dcell_model = make_dcellnet_for_label(true_model, label=TRUE_LABEL, 
                            correlation_data_file=corr_path,
                            p_threshold=CLUSTER_THRESHOLD, 
                            min_cluster_size=MIN_CLUSTER_SIZE,
                            plot_ontology=False)
    
    ''' Load or train DCell-like model on synthetic data'''
    dcell_path = WORKING_DIR + DCELL_MODEL_NAME
    if os.path.isfile(dcell_path):
        print('Loading DCell model...')
        dcell_model.load_state_dict(torch.load(dcell_path))
    else:
        print('Generating DCell test data (WT, all single KOs, some double KOs)...')
        train_dataset, test_dataset, train_labels, test_labels = \
            generate_dcell_train_data(BASE_IMAGE_HEX, true_model, 
                double_train_count=TRAINING_DOUBLE_COUNT,
                double_test_count=VALIDATION_DOUBLE_COUNT, 
                seed=TEST_DATA_SEED)
        print('Training DCell model on synthetic data...')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
        dcell_model = train_epochs(dcell_model, train_loader, test_loader, epochs=DCELL_MODEL_EPOCHS)
        torch.save(dcell_model.state_dict(), dcell_path)
    dcell_model.eval()
    
    ''' Evaluate DCell-like model on higher order knockouts '''
    for i in range(2,MAX_EVAL_KNOCKOUT+1):
        print('Testing', str(i)+'-knockouts...')
        eval_dataset, eval_labels = generate_dcell_eval_data(BASE_IMAGE_HEX, 
            true_model, order=i, count=EVALULATION_COUNT, seed=TEST_DATA_SEED)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=TEST_BATCH_SIZE)
        test_model(dcell_model, eval_loader)
    
def train_true_model(model, input_batch=TRAIN_BATCH_SIZE, test_batch=TEST_BATCH_SIZE, 
                     lr=LEARNING_RATE, momentum=MOMENTUM, epochs=10, print_interval=25):
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
    model = train_epochs(model, train_loader, test_loader, lr, momentum, epochs, print_interval)
    return model

def train_epochs(model, train_loader, test_loader, lr=LEARNING_RATE,
                 momentum=MOMENTUM, epochs=10, print_interval=25):
    ''' Train a torch model using SGD '''
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)# weight_decay=WEIGHT_DECAY)
    for epoch in range(1, epochs + 1):
        train_model(model, train_loader, optimizer, epoch, print_interval)
        test_model(model, test_loader)
    return model

class LoggingPrinter:
    ''' Used to simultaneously print to file and console '''
    #https://stackoverflow.com/questions/24204898/python-output-on-both-console-and-file
    def __init__(self, filename):
        self.out_file = open(filename, "a+")
        self.old_stdout = sys.stdout #this object will take over `stdout`'s job
        sys.stdout = self #executed when the user does a `print`
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)
    def __enter__(self):  #executed when `with` block begins
        return self
    def __exit__(self, type, value, traceback):  #executed when `with` block ends
        #we don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout

if __name__ == '__main__':
    main()
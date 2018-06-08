#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:58:27 2018

@author: jhyun95
"""

import os, sys, datetime
import torch

from models import ConvNet, make_dcellnet_for_label, \
    train_true_model, train_dcell_model, test_dcell_model_single
from interactions import find_pixel_correlations, __get_prediction__
from data_generator import generate_dcell_train_data, generate_dcell_eval_data

''' Global pipeline parameters '''
WORKING_DIR = '../data/DCell_test/'
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.05
MOMENTUM = 0.5

''' Images most like real cell viability model '''
WT_TEST = '0x1e000003e000007e000007e000007c000003c000000f8000000f0000003c000000600000038000001c000000e00000070001f8380007ff80001ff8000000000000000000000000000000000000000000'
WT_TEST_LABEL = 5 # Image should be 5, ~14k positive interactions, ~140k negative interactions
WT_A = '0x3000000f800000fc000007c000003c000003c000003c000007c000007c00000f800000f800000f800000f800000f000000f000001e000001e000001e000001c0000018000000000000000000000000'
WT_B = '0x30000003800001b8000039c000071c0000f1c0001f9c0001ffc000003c000001c000001c000001c000000e000000c000000e000000e000000e000000e000000e00000020000000000000000'
WT_C = '0x1ac00003fe00003fe00007f8000077c00007fe00007ff0000f8f000040780000038000003c000001c0000018000c038000e0f0000f3f0000ffe00007fc00007f000003c000000000000000000000000'
WT_D = '0x180000018000003800003f000003e000003e000003c000007c000007e00000ee00001c6000038700007070000e070001c078001c070003c0f0003fff0003ffc0001ff000000000000000000000000000000000000000000000'
WT_E = '0x60000007000000f000001fc00007ff8000f87c001f01e003e007003c00300380030000006000000e000000e000fc1c001ffb8003bff000383fc001fffc000ff180003c00000000000000000000000000000000000000'
WT_A_LABEL = 1
WT_B_LABEL = 4
WT_C_LABEL = 5
WT_D_LABEL = 0
WT_E_LABEL = 2

def main():
    ''' Initialize working directory and log '''
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    log_file = WORKING_DIR + 'log_min1.txt'
    
    ''' Print to console and log file '''
    with LoggingPrinter(log_file):
        print('------ BEGIN LOG:', datetime.datetime.now(), '-----------------------------------')
        true_model = fit_true_model(model_name='ConvNet_E20', epochs=20)
#        test_interactions(true_model) # only need to run this once per true model
#        benchmark_dcell(true_model, WT_TEST, WT_TEST_LABEL, model_name='DCell_test')
        benchmark_dcell(true_model, WT_A, WT_A_LABEL, model_name='DCell_A')
        benchmark_dcell(true_model, WT_B, WT_B_LABEL, model_name='DCell_B')
        benchmark_dcell(true_model, WT_C, WT_C_LABEL, model_name='DCell_C')
        benchmark_dcell(true_model, WT_D, WT_D_LABEL, model_name='DCell_D')
        benchmark_dcell(true_model, WT_E, WT_E_LABEL, model_name='DCell_E')
        
def fit_true_model(model_name, epochs):
    ''' Load or train the "true" image model '''
    model_path = WORKING_DIR + model_name
    true_model = ConvNet()
    if os.path.isfile(model_path):
        print('Loading true image model...')
        true_model.load_state_dict(torch.load(model_path))
    else:
        print('Training true image model...')
        true_model = train_true_model(true_model, epochs=epochs)
        torch.save(true_model.state_dict(), model_path)
    true_model.eval()
    return true_model

def test_interactions(true_model):
    ''' Test for single and pairwise pixel interactions, for wildtype
        image selection. '''
    from interactions import find_lethal_pixels, find_2nd_order_interactions
    FIRST_ORDER_FILE = WORKING_DIR + '1st_order.tsv'
    SECOND_ORDER_FILE = WORKING_DIR + '2nd_order.tsv'
    find_lethal_pixels(true_model, output_file=FIRST_ORDER_FILE)
    find_2nd_order_interactions(true_model, first_order_file=FIRST_ORDER_FILE,
                                output_file=SECOND_ORDER_FILE)

def benchmark_dcell(true_model, wt_image_hex, wt_label, model_name='DCell_E1'):
    ''' Benchmark DCell against a true model and a selected wildtype image '''
    CORRELATION_MODE = 'mcc-adj'
    DCELL_MODEL_EPOCHS = 2 # number of epochs to train the DCell model
    TEST_DATA_SEED = 1 # randomization seed for generating synthetic data
    CLUSTER_THRESHOLD = 0.05 # p-value cutoff of hierarchical clustering
    MIN_CLUSTER_SIZE = 10 # minimum cluster size for hierarchical clustering
    TRAINING_DOUBLE_COUNT = 100000 # number of double KOs to train DCell with
    VALIDATION_DOUBLE_COUNT = 20000 # not that important, quick check for robustness
    MAX_EVAL_KNOCKOUT = 15 # evaluate up to KOs of this size
    EVALULATION_COUNT = 100000 # number of higher order KOs to test per size
    MIN_NEURONS_PER_TERM = 5 # default is 15
    PLOT_ONTOLOGY = False # plot the DCell ontology
       
    print('Base Image:', wt_image_hex)
    print('Base Image Label:', wt_label)
    print('Test Data Seed:', TEST_DATA_SEED)
        
    ''' Analyze base image with respect to the true model '''
    base_label = __get_prediction__(true_model, wt_image_hex)
    if base_label != wt_label:
        print('WARNING: True model incorrectly labels the base image')
        sys.exit(1)
    else:
        print('True model and label agree for base image:', wt_label)
    
    ''' From target label, pre-compute pixel correlations for DCell model '''
    corr_path = WORKING_DIR + 'pixel-' + str(wt_label) +  \
                    '-' + CORRELATION_MODE + '.csv.gz'
    if not os.path.isfile(corr_path):
        find_pixel_correlations(true_model, labelset=wt_label, 
                                check_consistency=True, mode=CORRELATION_MODE, 
                                output_file=corr_path)
    
    ''' Initialize DCell-like network from correlations for image label '''
    print('Initializing DCell network...')
    dcell_model = make_dcellnet_for_label(true_model, label=wt_label, 
                            correlation_data_file=corr_path,
                            p_threshold=CLUSTER_THRESHOLD, 
                            min_cluster_size=MIN_CLUSTER_SIZE,
                            plot_ontology=PLOT_ONTOLOGY, 
                            min_neurons_per_term=MIN_NEURONS_PER_TERM)
    
    ''' Load or train DCell-like model on synthetic data'''
    dcell_path = WORKING_DIR + model_name
    if os.path.isfile(dcell_path):
        print('Loading DCell model...')
        dcell_model.load_state_dict(torch.load(dcell_path))
    else:
        print('Generating DCell test data (WT, all single KOs, some double KOs)...')
        train_dataset, test_dataset, train_labels, test_labels = \
            generate_dcell_train_data(wt_image_hex, true_model, 
                double_train_count=TRAINING_DOUBLE_COUNT,
                double_test_count=VALIDATION_DOUBLE_COUNT, 
                seed=TEST_DATA_SEED)
        print('Training DCell model on synthetic data...')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
        dcell_model = train_dcell_model(dcell_model, train_loader, test_loader, epochs=DCELL_MODEL_EPOCHS)
        torch.save(dcell_model.state_dict(), dcell_path)
    dcell_model.eval()
    
    ''' Evaluate DCell-like model on higher order knockouts '''
    for i in range(2,MAX_EVAL_KNOCKOUT+1):
        print(model_name + ': Testing', str(i)+'-knockouts...')
        eval_dataset, eval_labels = generate_dcell_eval_data(wt_image_hex, 
            true_model, order=i, count=EVALULATION_COUNT, seed=TEST_DATA_SEED)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=TEST_BATCH_SIZE)
        eval_corr, eval_acc = test_dcell_model_single(dcell_model, eval_loader)
        print('MCC:', eval_corr)
        print('ACC:', eval_acc)

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
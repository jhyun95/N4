# -*- coding: utf-8 -*-
"""
Created on Mon May 28 00:40:08 2018

@author: jhyun_000
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from interactions import DIM, __get_pixel__, __apply_corruptions__, \
    __hex_to_image__, __get_prediction__
    
def generate_dcell_eval_data(base, true_model, order, count=100000, seed=1):
    ''' Generates evaluation data for DCell testing model  as TorchDatasets.
        Intended to simulate higher-order knockouts that would normally
        not be measured. '''
    knockouts = generate_random_pixel_groups(size=order, count=count, seed=seed)
    print('Generating tensors from evaluation knockouts...')
    dataset, labels = convert_knockouts_to_tensor_dataset(base, true_model, knockouts)
    return dataset, labels

def generate_dcell_train_data(base, true_model, double_train_count=100000, 
                              double_test_count=20000, seed=1):
    ''' Generates training and testing/validation data for DCell testing model 
        as torch TensorDatasets (already shaped to work in DCellNet). Includes:
        - All single "knockouts", when a single pixel is flipped 
        - A number of double "knockouts", when two pixels are flipped '''
    total_doubles = double_train_count + double_test_count
    wildtype = [ tuple() ]
    singles = generate_random_pixel_groups(size=1, count=DIM*DIM, seed=seed)
    doubles = generate_random_pixel_groups(size=2, count=total_doubles, seed=seed)
    train_doubles = doubles[:double_train_count]
    test_doubles = doubles[double_train_count:]
    
    train_set = wildtype + singles + train_doubles
    test_set = test_doubles
    print('Generating tensors from training knockouts...')
    train_dataset, train_labels = convert_knockouts_to_tensor_dataset(base, true_model, train_set)
    print('Generating tensors from testing knockouts...')
    test_dataset, test_labels = convert_knockouts_to_tensor_dataset(base, true_model, test_set)
    
    return train_dataset, test_dataset, train_labels, test_labels

def convert_knockouts_to_tensor_dataset(base, true_model, knockouts, print_counter=5000):
    ''' Takes a list of lists, where each sublist corresponds to a set 
        of pixels to flip or "knockout" relative to a base image. 
        Map each knockout to a tensor to create TensorDataset '''
    if type(base) == str: # hexstring of image provided
        image = __hex_to_image__(base)
    else: # image tensor provided, shape to DIMxDIM
        image = base.view(1,1,DIM,DIM)
        
    ''' Generate feature and target tensors '''
    labels = {}
#    features = torch.zeros([len(knockouts), DIM, DIM]).byte()
    features = torch.zeros([len(knockouts), DIM*DIM]).float()
    targets = torch.zeros(len(knockouts)).long()
    for i in range(len(knockouts)):
        if i % print_counter == 0:
            print('On image', i, 'of', len(knockouts))
        pixel_indices = knockouts[i]
        pixels = map(__get_pixel__, pixel_indices)
        corrupted = __apply_corruptions__(image, pixels)
        target = __get_prediction__(true_model, corrupted)
#        features[i] = corrupted.data.byte()
#        targets[i] = target.data.byte()[0]
        features[i] = corrupted.view(1,1,DIM*DIM).data.float()
        targets[i] = target.data.float()[0]
        labels[knockouts[i]] = target.data.byte()[0]
        
    return TensorDataset(features, targets), labels

def generate_random_pixel_groups(size, count, seed=1):
    ''' Generates random pixel group without replacement. Returns a list of 
        tuples, each comprised with a set of pixel indices from 0 to DIM*DIM.
        For singles and doubles, all choices are enumerated then shuffled. 
        For larger groups, choices are drawn randomly and repeats are 
        dropped, until the desired count is reached. '''
    np.random.seed(seed=seed); output = []
    if size == 1: # for singles, enumerate all choices and shuffle
        singles = np.arange(DIM*DIM)
        np.random.shuffle(singles)
        for i in range(count):
            output.append( (singles[i],) )
        return output
    elif size == 2: # for pairs, enumerate all pairs and shuffle
        double_ko_choices = int(DIM*DIM*(DIM*DIM-1) / 2)
        double_ko_pairs = np.zeros((double_ko_choices,2), dtype=np.int)
        counter = 0
        for i in range(DIM*DIM):
            for j in range(i):
                double_ko_pairs[counter,0] = i
                double_ko_pairs[counter,1] = j
                counter += 1
        np.random.shuffle(double_ko_pairs)
        for i in range(count):
            output.append(tuple(double_ko_pairs[i,:]))
        return output
    elif size > 2: # for larger groups, generate repeatedly new random groups
        choices = set()
        while len(choices) < count:
            group = tuple(np.random.randint(0,DIM*DIM,size=size))
            if len(group) == len(set(group)): # pixels are unique
                choices.add(group)
        return list(choices)
    return []
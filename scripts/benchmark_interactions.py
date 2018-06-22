# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:15:11 2018

@author: jhyun_000
"""

import itertools, time, datetime
import numpy as np
import torch

from models import ConvNet
from interactions import DIM, __get_pixel__
from data_generator import convert_knockout_to_tensor, generate_random_pixel_groups
from benchmark import LoggingPrinter, WT_A, WT_B, WT_C, WT_D, WT_E
    
def main():
    wildtypes = [WT_A, WT_B, WT_C, WT_D, WT_E]
    labels = ['WT_A', 'WT_B', 'WT_C', 'WT_D', 'WT_E']
    with LoggingPrinter('../data/DCell_test/log_interactions.txt'):
        print('------ BEGIN LOG:', datetime.datetime.now(), '-----------------------------------')
        for wti in range(len(wildtypes)):
            wildtype = wildtypes[wti]
            label = labels[wti]
            print('Testing', label)
            true_model = ConvNet()
            true_model.load_state_dict(torch.load('../data/DCell_test/ConvNet_E20'))
            true_model.eval()
            for i in range(3,11):
                start_time = time.time()
                test_interactions(wildtype, true_model, order=i, count=50000, batch_size=1024)
                elapsed = time.time() - start_time
                print('Time (seconds):', elapsed)
    
def test_interactions(base, true_model, order=3, count=1024, starting_size=1,
                      batch_size=1024, seed=1):
    ''' Simulates knockouts of a particular order, and counts how many
        have a full interaction (i.e. for a 5-KO, if there is fifth order 
        interaction). This is defined as:
        - If at least one subset KO is lethal but the full KO is nonlethal,
          then there is a full positive interaction 
        - If all subset KOs are nonlethal but the full KO is lethal,
          then there is a full negative interaction '''
    print('Testing', count, str(order)+'-KOs', 'for interactions...')
    true_model.eval()
    knockouts = generate_random_pixel_groups(order, count, seed=seed)
    base_image, base_label = convert_knockout_to_tensor(base, true_model, [])
    base_array = base_image.view(DIM,DIM).data.numpy().astype(bool)
    
    positive_interactions = 0
    negative_interactions = 0
    progress = 0
    for i in range(0, count, batch_size): # break up knockouts into minibatches
        minibatch = np.array(knockouts[i:i+batch_size])
        n = len(minibatch); progress += n
        has_lethal_subset = np.zeros(n, dtype=bool)
        all_subsets_nonlethal = np.zeros(n,dtype=bool) + 1
        
        ''' Test lethalithy of subknockouts and full knockout '''
        for sub_ko_size in range(starting_size,order+1): # generate sub-knockouts 
            for sub_ko_indices in itertools.combinations(range(order), sub_ko_size):
                ''' Generate subknockouts in numpy '''
                subknockouts = minibatch[:,sub_ko_indices] # sub-knockout pixel indices
                sub_ko_mask = np.zeros((n, DIM, DIM)) # convert to sub-knockout pixel masks
                for j in range(n):
                    for k in subknockouts[j]:
                        x,y = __get_pixel__(k)
                        sub_ko_mask[j,x,y] = 1
                sub_ko_images = np.zeros((n, DIM, DIM)) # convert masks to actual images
                sub_ko_images[:] = base_array # initialize each image as wildtype
                sub_ko_images = np.logical_xor(sub_ko_images, sub_ko_mask) # apply mask
                
                ''' Convert subknockouts to torch and run through model '''
                sub_ko_images = sub_ko_images.astype(float) # convert to float for torch
                sub_ko_tensors = torch.FloatTensor(sub_ko_images).view(n,1,DIM,DIM) # convert to torch tensor
                output = true_model(sub_ko_tensors)
                weight, pred = torch.max(output, 1)
                nonlethal = (pred == base_label).data.numpy().astype(bool)
                lethal = np.logical_not(nonlethal)
                
                if sub_ko_size < order: # testing subknockout
                    has_lethal_subset = np.logical_or(has_lethal_subset, lethal)
                    all_subsets_nonlethal = np.logical_and(all_subsets_nonlethal, nonlethal)
                else: # testing full knockout
                    full_is_lethal = lethal; full_is_nonlethal = nonlethal
        
        ''' Test for positive or negative interaction'''
        has_positive_interaction = np.logical_and(full_is_nonlethal, has_lethal_subset)
        has_negative_interaction = np.logical_and(full_is_lethal, all_subsets_nonlethal)
        positive_interactions += np.count_nonzero(has_positive_interaction)
        negative_interactions += np.count_nonzero(has_negative_interaction)
        print('Tested', progress, 'of', str(count)+'; PI/NIs:', positive_interactions, negative_interactions)
    
    lethal_count = np.count_nonzero(full_is_lethal)
    lethal_rate = round(lethal_count / count * 100, 5)
    pi_rate = round(positive_interactions / count * 100, 5)
    ni_rate = round(negative_interactions / count * 100, 5)
    print('> Total Lethal:', lethal_count, '(' + str(lethal_rate) + '%)')
    print('> Positive Interactions:', positive_interactions, '(' + str(pi_rate) + '%)')
    print('> Negative Interactions:', negative_interactions, '(' + str(ni_rate) + '%)')
        

if __name__ == '__main__':
    main()
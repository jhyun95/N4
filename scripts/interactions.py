# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:52:13 2018

@author: jhyun_000
"""

from __future__ import print_function
import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import models
import random, time, os, multiprocessing

torch.manual_seed(1) # FIXED SEED FOR REPRODUCIBILITY
DIM = 28 # images are 28x28
TOTAL_IMAGES = 10000 # total number of images
MODEL = models.BaseNet()
MODEL.load_state_dict(torch.load("../models/BaseNet_E10"))
OUTPUT_1ST = '../data/1st_order_seed_1.tsv'
OUTPUT_2ND = '../data/2nd_order_seed_1.tsv'
CORR_DIR = '../data/pixel_correlations/'

def main():
#    third_order_test()
#    find_lethal_pixels()
#    find_2nd_order_interactions()
    
    compute_pixel_correlations_parallel(mode='mcc-adj')
    
#    heatmap_correlation(CORR_DIR + 'pixel_sokal_michener_0s.csv.gz'); plt.figure()
#    heatmap_correlation(CORR_DIR + 'pixel_mcc_0s.csv.gz'); plt.figure()
#    heatmap_correlation(CORR_DIR + 'pixel_mcc_adj_0s.csv.gz')
    
def compute_pixel_correlations_parallel(sets=range(10), mode='mcc-adj', cpus=2):
    ''' Compute pixel correlations for a given mode and all labelsets,
        based on find_pixel_correlations '''
    pool = multiprocessing.Pool(cpus)
    pool_data = map(lambda i: (i,mode), sets)
    pool.map(__pixel_correlation_helper__, pool_data)

def __pixel_correlation_helper__(i_mode):
    ''' Helper function for parallelized pixel correlation calculation '''
    i,mode = i_mode
    filename = CORR_DIR + 'pixel_' +mode.replace('-','_') 
    filename += '_' + str(i) + 's.csv.gz'
    find_pixel_correlations(labelset=i, mode=mode, output_file=filename)
    
def heatmap_correlation(data_file):
    ''' Heatmap of pixel correlation data '''
    data = np.loadtxt(data_file, delimiter=',')
    sns.heatmap(data, vmin=-1, vmax=1, cmap='RdBu')
    
def get_prediction(model, data):
    ''' Gets the predicted number for an image represented as a 1x1xDIMxDIM tensor'''
    output = model(data)
    weight, pred = torch.max(output,1)
    
def apply_corruptions(data, pixels):
    ''' Flips pixels in an image represented as a 1x1xDIMxDIM tensor '''
    mod_data = data.clone()
    for r,c in pixels:
        mod_data[0,0,r,c] = 1 - data[0,0,r,c]
    return mod_data

def image_to_hex(data):
    ''' Converts 28x28 binary image to hex string '''
    collapsed = data.view(DIM*DIM) # convert to 1D
    bits = map(lambda px: str(int(px)), collapsed) # convert to 0s and 1s
    bitstring = ''.join(list(bits)) # concatenate to bitstring
    return hex(int(bitstring,2)) # return hex object

def hex_to_image(hexstring):
    ''' Converts hexstring to 28x28 binary image (1x1x28x28 tensor) '''
    bitstring = bin(int(hexstring,16))[2:] # convert to binary string
    bitstring = '0' * (DIM*DIM-len(bitstring)) + bitstring # pad zeros
    bitstring = list(map(int, list(bitstring)))
    data = torch.FloatTensor(list(bitstring))
    data = data.view(1,1,DIM,DIM)
    data = torch.autograd.Variable(data, volatile=True)
    return data

def get_pixel(p):
    ''' Maps integer from 0 to DIM*DIM-1 to pixel coordinates '''
    return (p % DIM, int((p-p%DIM)/DIM))

def find_pixel_correlations(model=MODEL, labelset=2, check_consistency=True,
                            mode='sokal-michener', output_file='pixel_corr.csv'):
    ''' Create pairwise pixel correlations for either all images, or for 
        a particular label (i.e. 2s only). Uses model-generated labels, 
        not actual labels. Checks for consistency by default, i.e. 
        model labels = actual labels.
        
        Refer to http://www.iiisci.org/journal/CV$/sci/pdfs/GS315JG.pdf for 
        more possible metrics for correlating binary variables:
            sokal-michener = 2 * # matches / # images - 1.0 (scaled onto [-1,1])
            mcc = matthews correlation coefficient (equivalent to pearson)
            mcc-adj = mcc, but add 1 to each FP, FN, TP, TN to avoid 0 denom
    '''
        
    ''' First find images that match the desired label '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(models.transform_list())),
        batch_size=1, shuffle=False)  
        
    model.eval(); images = []
    start_time = time.time()
    print('Loading images that match:', labelset)
    for data, target in test_loader:
        data = torch.autograd.Variable(data, volatile=True)
        target = torch.autograd.Variable(target)
        output = model(data)
        weight, base_pred = torch.max(output, 1)
        if labelset == None or int(base_pred) == labelset:
            if int(target) == labelset or not check_consistency:
                images.append(data.view(DIM,DIM)) # flatted to 2D tensor
    N = len(images)
    print('Loaded', N, 'images.')
    images = torch.stack(images)
    
    ''' Compute pairwise pixel correlations for black and white images
        NOTE: Simply takes the 2*(# times pixels equal) / (# images) - 1'''    
    correlations = np.zeros(shape=(DIM*DIM,DIM*DIM))
    for p1 in range(DIM*DIM):
        print('Image Set:', labelset, 'Pixel #:', p1+1, 'of', DIM*DIM)        
        correlations[p1,p1] = 1.0 # self correlation
        x1,y1 = get_pixel(p1)
        px1data = images[:,x1,y1].data.cpu().numpy().astype(bool)
        for p2 in range(p1):
            x2,y2 = get_pixel(p2)
            px2data = images[:,x2,y2].data.cpu().numpy().astype(bool)
            if mode == 'sokal-michener':  
                mismatches = np.sum(np.logical_xor(px1data, px2data), dtype=np.int64)
                matches = N - mismatches
                corr = 2.0 * matches / N - 1.0
            elif mode == 'mcc' or mode == 'mcc-adj':
                a = np.sum(np.logical_and(px1data, px2data), dtype=np.int64) # both white
                b = np.sum(px1data, dtype=np.int64) - a # white/black
                c = np.sum(px2data, dtype=np.int64) - a # black/white
                d = N - a - b - c # both black
                if mode == 'mcc-adj': # pseudocounts
                    a += 1; b += 1; c += 1; d += 1
                numer = (a+d) - (b+c)
                denom = (a+b)*(a+c)*(b+d)*(c+d)
                corr = numer/denom**0.5 if denom > 0 else 0.0
            correlations[p1,p2] = corr
            correlations[p2,p1] = corr
    
    np.savetxt(output_file, correlations, delimiter=',', newline='\n')
    print('Time (seconds):', round(time.time() - start_time, 3))

def find_2nd_order_interactions(model=MODEL, first_order_file=OUTPUT_1ST,
                                output_file = OUTPUT_2ND):
    ''' For all images that have lethal pixels and are consistent (base 
        prediction = actual label), scan pairs of pixels for interactions:
        - If either pixel is lethal, but the combined is not = Positive 
        - If neither pixel is lethal, but combined is = Negative '''
    
    ''' Extract only consistent images with 1st-order interactions '''
    f = open(first_order_file,'r+')
    f.readline() # skip header
    consistent_images = []; inconsistent_images = []
    lethal_pixels = {}; base_identity = {}
    for line in f:
        hexstring, target, base, lethal = line.split('\t')
        base_identity[hexstring] = (int(target), int(base))
        lethal_pixels[hexstring] = []
        for entry in lethal.split(';'):
            r,c = entry.strip().split(',')
            r = int(r[1:]); c = int(c[:-1])
            lethal_pixels[hexstring].append((r,c))
        if target == base: # consistent, prediction = label
            consistent_images.append(hexstring) 
        else: # inconsistent, prediction = label
            inconsistent_images.append(hexstring)
    print('Loaded', len(consistent_images), 'consistent images with lethal pixels.')
    print('Loaded', len(inconsistent_images), 'inconsistent images with lethal pixels.')
    
    ''' Iterate through pixel pairs for 2nd-order interactions '''
    already_run = set()
    if not os.path.isfile(output_file): # initialize if not existent
        f = open(output_file, 'w+')
        f.write('image_hex_string\tactual\tbase_prediction\tpositive_interactions\tnegative_interactions')
        f.close()
    else: # file already exists
        f = open(output_file, 'r+')
        f.readline()
        for line in f:
            hexstring = line.split('\t')[0]
            already_run.add(hexstring)
        f.close()
    
    image_counter = 1
    total_pairs = 306936.0
    for hexstring in consistent_images:
        base = base_identity[hexstring][1]
        print('Image', image_counter, '(' + str(base) + ')', hexstring)
        if not hexstring in already_run: # check if there is already an entry for this image
            start_time = time.time()
            data = hex_to_image(hexstring) # to pytorch tensor
            positive_interactions = []; negative_interactions = []
            for i in range(DIM*DIM): # pixel 1
                px1 = get_pixel(i); is_lethal1 = px1 in lethal_pixels[hexstring]
                for j in range(i): # pixel 2
                    px2 = get_pixel(j); is_lethal2 = px2 in lethal_pixels[hexstring]
                    corrupted = apply_corruptions(data, [px1, px2])
                    double_swap = get_prediction(model, corrupted)
                    is_lethal12 = double_swap != base
                    if is_lethal1 or is_lethal2: # either one is lethal
                        if not is_lethal12: # but both is not = POSITIVE interaction
                            positive_interactions.append((px1,px2))
        #                        print('POSITIVE:', px1, px2, is_lethal1, is_lethal2, is_lethal12)
                    else: # neither is lethal
                        if is_lethal12: # both is = NEGATIVE interactions
                            negative_interactions.append((px1,px2))
        #                        print('NEGATIVE:', px1, px2, is_lethal1, is_lethal2, is_lethal12)
            positives = len(positive_interactions); percent_positive = round(positives/total_pairs,3)
            negatives = len(negative_interactions); percent_negative = round(negatives/total_pairs,3)
            print('Positive Interactions:', positives,  '(' + str(percent_positive) + '%)')
            print('Negative Interactions:', negatives,  '(' + str(percent_negative) + '%)')
            print('Time (seconds):', round(time.time() - start_time, 3))
            # Append to output file
            f = open(output_file, 'a+')
            out = hexstring + '\t' + str(base) + '\t' + str(base) + '\t' 
            out +=str(positives) + '\t' + str(negatives) + '\n'
            f.write(out)
            f.close()
        else: # detected in output file / already run
            print('Already run, found in', output_file)
        image_counter += 1

def find_lethal_pixels(model=MODEL, output_file=OUTPUT_1ST):
    ''' For each image, swap every pixel one at a time to see 
        if the prediction is altered / image is corrupted '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(models.transform_list())),
        batch_size=1, shuffle=False)  
    
    model.eval(); image_counter = 0
    lethal_pixels = {}
    for data, target in test_loader:
        image_counter += 1
        data = torch.autograd.Variable(data, volatile=True)
        target = torch.autograd.Variable(target)
        image_hex = image_to_hex(data)
        output = model(data)
        weight, base_pred = torch.max(output, 1)
        target = int(target); base_pred = int(base_pred)
        print('Image', image_counter, 'Label:', target,  'Predicted:', base_pred)
        for px in range(DIM*DIM):
            pixel = get_pixel(px)
            corrupted = apply_corruptions(data, [pixel])
            pred = get_prediction(model, corrupted)
            if pred != base_pred: # if image was corrupted
                print('LETHAL:', image_counter, base_pred, pred, pixel)
                if not image_hex in lethal_pixels:
                    lethal_pixels[image_hex] = [target, base_pred, []]
                lethal_pixels[image_hex][-1].append(pixel)
    
    f = open(output_file,'w+')
    f.write('image_hex_string\tactual\tbase_prediction\tcorrupting_pixels')
    for image_hex in lethal_pixels:
        target, base, pixels = lethal_pixels[image_hex]
        output = image_hex + '\t' + str(target) + '\t' + str(base) 
        output += '\t' + ';'.join(map(str,pixels))
        f.write('\n' + output) 
    f.close()

def third_order_test(model=MODEL):
    ''' Randomly test images for threeway interactions '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(models.transform_list())),
        batch_size=1, shuffle=False)

    model.eval()
    image_counter = 0
    for data, target in test_loader:
        image_counter += 1
        data = torch.autograd.Variable(data, volatile=True)
        target = torch.autograd.Variable(target)
        output = model(data)
        weight, base_pred = torch.max(output, 1)
        print('Image', image_counter, 'Label:', int(target),  'Predicted:', int(base_pred))
        for limit in range(100):
            if (limit+1) % 1000 == 0:
                print('\tTesting image', image_counter, 'triple', limit+1)
            p1 = random.randint(0,28*28-1)
            p2 = random.randint(0,28*28-1)
            p3 = random.randint(0,28*28-1)
            if p1 != p2 and p1 != p3 and p2 != p3:
                results = test_threeway(model, data, p1, p2, p3)
                corrupted = len(set(results.values())) > 1 # more than one predicted output
                if corrupted:
                    print(results)
                    
def test_threeway(model,data,p1,p2,p3):
    ''' Test all possible perturbations based on 3 pixels '''
    pixel_sets = [ [], [p1], [p2], [p3], [p1,p2], [p1,p3], [p2,p3], [p1,p2,p3] ]
    predictions = {}
    for pixel_set in pixel_sets:
        pixels = tuple(map(get_pixel, pixel_set))
        corrupted = apply_corruptions(data, pixels)
        predictions[pixels] = get_prediction(model, corrupted)
    return predictions
                    
if __name__ == '__main__':
    main()
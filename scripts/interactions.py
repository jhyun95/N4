# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:52:13 2018

@author: jhyun_000
"""

from __future__ import print_function
import torch
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.misc
import random, time, os, multiprocessing

from models import transform_black_and_white, ConvNet

torch.manual_seed(1) # FIXED SEED FOR REPRODUCIBILITY
DIM = 28 # images are 28x28
TOTAL_IMAGES = 10000 # total number of images
OUTPUT_1ST = '../data/1st_order_seed_1.tsv'
OUTPUT_2ND = '../data/2nd_order_seed_1.tsv'
CORR_DIR = '../data/pixel_correlations/'

def main():
    ''' Examines 1st order and 2nd order interactions, and also computes
        pixel correlations when taking ConvNet as the true model. '''
    model = ConvNet()
    model.load_state_dict(torch.load("../models/ConvNet_E10"))
    
    ''' Old exploratory functions '''
#    third_order_test(model)
#    find_lethal_pixels(model)
#    find_2nd_order_interactions(model)

    ''' Pixel Correlation Calculation '''
    
#    find_pixel_correlations_parallel(model, sets=range(10), mode='sokal-michener', cpus=2)
#    find_pixel_correlations_parallel(model, sets=range(10), mode='mcc', cpus=2)
#    find_pixel_correlations_parallel(model, sets=range(10), mode='mcc-adj', cpus=2)
    
#    heatmap_correlation(CORR_DIR + 'pixel_sokal_michener_0s.csv.gz'); plt.figure()
#    heatmap_correlation(CORR_DIR + 'pixel_mcc_0s.csv.gz'); plt.figure()
#    heatmap_correlation(CORR_DIR + 'pixel_mcc_adj_0s.csv.gz')

def generate_dcell_knockout_data(base, true_model, order=2, count=10000,
                                 flatten=True, seed=1):
    ''' Generates a fixed number of unique knockouts of a certain order.
        Note: May be slow if count ~ number of possible knockouts for order > 2 '''
    if type(base) == str: # hexstring of image provided
        image = __hex_to_image__(base)
    else: # image tensor provided, shape to DIMxDIM
        image = base.view(DIM,DIM,1,1)
    
    ''' Initialize feature and targets tensor '''
    max_count = scipy.misc.comb(N=DIM*DIM, k=order)
    feature_count = min(max_count, count)
    features = torch.zeros([feature_count, DIM, DIM]).byte()
    targets = torch.zeros(feature_count).byte()
    
    ''' Generate knockouts '''
    if order == 0: # wildtype
        target = __get_prediction__(true_model, image)
        features[0] = image.data.byte()
        targets[0] = target.data.byte()[0]
    else: # knockout
        knockouts = __generate_random_pixel_groups__(order, count, seed)
        for i in range(count):
            px = knockouts[i]
            pixels = map(__get_pixel__, px)
            corrupted = __apply_corruptions__(image, pixels)
            target = __get_prediction__(true_model, corrupted)
            features[i + DIM*DIM] = corrupted.data.byte()
            targets[i + DIM*DIM] = target.data.byte()[0]
        
    return TensorDataset(features, targets)
    
def generate_dcell_data(base, true_model, 
                        double_train_count=100000, 
                        double_test_count=20000,
                        flatten=True, seed=1):
    ''' Generates training data for DCell testing model as a torch Dataset.
        Uses ByteTensors to save memory, transform to FloatTensor before 
        feeding into DCellNet via a DataLoader. Includes:
        - All single "knockouts", when a single pixel is flipped 
        - A number of double "knockouts", when two pixels are flipped '''
    total_doubles = double_train_count + double_test_count
    
    wildtype = generate_dcell_knockout_data(
            base, true_model, order=0, count=1, flatten, seed)
    singles = generate_dcell_knockout_data(
            base, true_model, order=1, count=DIM*DIM, flatten, seed)
    doubles = generate_dcell_knockout_data(
            base, true_model, order=2, count=total_doubles, flatten, seed)
    train_doubles = Subset(doubles, np.arange(train_count))
    test_doubles = Subset(doubles, np.arange(train_count, total_doubles))
    
    train_set = ConcatDataset([wildtype, singles, train_doubles])
    return train_set, test_doubles
    
def heatmap_correlation(correlation_data_file):
    ''' Heatmap of pixel correlation data '''
    data = np.loadtxt(correlation_data_file, delimiter=',')
    sns.heatmap(data, vmin=-1, vmax=1, cmap='RdBu')
    
def find_pixel_correlations_parallel(model, sets=range(10), mode='mcc-adj', cpus=2):
    ''' Compute pixel correlations for a given mode and all labelsets,
        based on find_pixel_correlations. NOTE: Currently not functional, 
        cannot pickle model for parallelization. Use serial version. '''
    def __pixel_correlation_helper__(i_mode):
        ''' Helper function for parallelized pixel correlation calculation '''
        i,mode = i_mode
        filename = CORR_DIR + 'pixel_' +mode.replace('-','_') 
        filename += '_' + str(i) + 's.csv.gz'
        find_pixel_correlations(model, labelset=i, mode=mode, output_file=filename)
        
    pool = multiprocessing.Pool(cpus)
    pool_data = map(lambda i: (i,mode), sets)
    pool.map(__pixel_correlation_helper__, pool_data)

def find_pixel_correlations(model, labelset=2, check_consistency=True,
                            mode='sokal-michener', output_file=None):
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
                       transform=transforms.Compose(transform_black_and_white())),
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
    
    ''' Compute pairwise pixel correlations for black and white images '''    
    correlations = np.zeros(shape=(DIM*DIM,DIM*DIM))
    for p1 in range(DIM*DIM):
        print('Image Set:', labelset, 'Pixel #:', p1+1, 'of', DIM*DIM)        
        correlations[p1,p1] = 1.0 # self correlation
        x1,y1 = __get_pixel__(p1)
        px1data = images[:,x1,y1].data.cpu().numpy().astype(bool)
        for p2 in range(p1):
            x2,y2 = __get_pixel__(p2)
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
                numer = (a*d) - (b*c)
                denom = (a+b)*(a+c)*(b+d)*(c+d)
                corr = (numer*numer/denom)**0.5 if denom > 0 else 0.0
                if np.abs(corr) > 1.0: # rounding error?
                    print('WARNING:',corr,a,b,c,d)
            correlations[p1,p2] = corr
            correlations[p2,p1] = corr
    
    if output_file != None:
        np.savetxt(output_file, correlations, delimiter=',', newline='\n')
    print('Time (seconds):', round(time.time() - start_time, 3))
    return correlations

def find_2nd_order_interactions(model, first_order_file=OUTPUT_1ST,
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
    total_pairs = DIM*DIM*(DIM*DIM-1)*0.5
    for hexstring in consistent_images:
        base = base_identity[hexstring][1]
        print('Image', image_counter, '(' + str(base) + ')', hexstring)
        if not hexstring in already_run: # check if there is already an entry for this image
            start_time = time.time()
            data = __hex_to_image__(hexstring) # to pytorch tensor
            positive_interactions = []; negative_interactions = []
            for i in range(DIM*DIM): # pixel 1
                px1 = __get_pixel__(i)
                is_lethal1 = px1 in lethal_pixels[hexstring]
                for j in __get_pixel__(i): # pixel 2
                    px2 = __get_pixel__(j)
                    is_lethal2 = px2 in lethal_pixels[hexstring]
                    corrupted = __apply_corruptions__(data, [px1, px2])
                    double_swap = __get_prediction__(model, corrupted)
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

def find_lethal_pixels(model, output_file=OUTPUT_1ST):
    ''' For each image, swap every pixel one at a time to see 
        if the prediction is altered / image is corrupted.
        Essentially a search for 1st order interactions '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(transform_black_and_white())),
        batch_size=1, shuffle=False)  
    
    model.eval(); image_counter = 0
    lethal_pixels = {}
    for data, target in test_loader:
        image_counter += 1
        data = torch.autograd.Variable(data, volatile=True)
        target = torch.autograd.Variable(target)
        image_hex = __image_to_hex__(data)
        output = model(data)
        weight, base_pred = torch.max(output, 1)
        target = int(target); base_pred = int(base_pred)
        print('Image', image_counter, 'Label:', target,  'Predicted:', base_pred)
        for px in range(DIM*DIM):
            pixel = __get_pixel__(px)
            corrupted = __apply_corruptions__(data, [pixel])
            pred = __get_prediction__(model, corrupted)
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
    
def __generate_random_pixel_groups__(size, count, seed=1):
    ''' Generates random pixel group without replacement.
        For singles and doubles, all choices are enumerated then shuffled. 
        For larger groups, choices are drawn randomly and repeats are 
        dropped, until the desired count is reached. '''
    np.random.seed(seed=seed)
    if size == 1: # for singles, enumerate all choices and shuffle
        singles = np.arange(DIM*DIM)
        np.random.shuffle(singles)
        return singles[:count]
    elif size == 2: # for pairs, enumerate all pairs and shuffle
        double_ko_choices = int(DIM*DIM*(DIM*DIM-1) / 2)
        double_ko_pairs = np.zeros((double_ko_choices,2), dtype=np.int8)
        counter = 0
        for i in range(DIM*DIM):
            for j in range(i):
                double_ko_pairs[counter,0] = i
                double_ko_pairs[counter,1] = j
                counter += 1
        np.random.shuffle(double_ko_pairs)
        return double_ko_pairs[:count]
    elif size > 2: # for larger groups, generate repeatedly new random groups
        choices = set()
        while len(choices) < count:
            group = tuple(np.random.random_integers(0,DIM*DIM-1,size))
            if len(group) == set(group): # pixels are unique
                choices.add(group)
        return np.array(list(choices))
    return np.array([])
    
def __get_prediction__(model, data):
    ''' Gets the predicted number for an image represented as a 1x1xDIMxDIM tensor'''
    output = model(data)
    weight, pred = torch.max(output,1)
    return pred
    
def __apply_corruptions__(data, pixels):
    ''' Flips pixels in an image represented as a 1x1xDIMxDIM tensor '''
    mod_data = data.clone()
    for r,c in pixels:
        mod_data[0,0,r,c] = 1 - data[0,0,r,c]
    return mod_data

def __image_to_hex__(data):
    ''' Converts 28x28 binary image to hex string '''
    collapsed = data.view(DIM*DIM) # convert to 1D
    bits = map(lambda px: str(int(px)), collapsed) # convert to 0s and 1s
    bitstring = ''.join(list(bits)) # concatenate to bitstring
    return hex(int(bitstring,2)) # return hex object

def __hex_to_image__(hexstring):
    ''' Converts hexstring to 28x28 binary image (1x1x28x28 tensor) '''
    bitstring = bin(int(hexstring,16))[2:] # convert to binary string
    bitstring = '0' * (DIM*DIM-len(bitstring)) + bitstring # pad zeros
    bitstring = list(map(int, list(bitstring)))
    data = torch.FloatTensor(list(bitstring))
    data = data.view(1,1,DIM,DIM)
    data = torch.autograd.Variable(data, volatile=True)
    return data

def __get_pixel__(p):
    ''' Maps integer from 0 to DIM*DIM-1 to pixel coordinates '''
    return (p % DIM, int((p-p%DIM)/DIM))

def third_order_test(model):
    ''' Randomly test images for threeway interactions.
        Initial exploratory approach. '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(transform_black_and_white())),
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
        pixels = tuple(map(__get_pixel__, pixel_set))
        corrupted = __apply_corruptions__(data, pixels)
        predictions[pixels] = __get_prediction__(model, corrupted)
    return predictions
                    
if __name__ == '__main__':
    main()
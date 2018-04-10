# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:52:13 2018

@author: jhyun_000
"""

from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
import random
from collections import Counter

MODEL= models.BaseNet()
MODEL.load_state_dict(torch.load("../models/BaseNet_E10"))

def test_threeway(model,data,p1,p2,p3):
    get_pixels = lambda p: (p % 28, int((p-p%28)/28))
    pixel_sets = [ [], [p1], [p2], [p3], [p1,p2], [p1,p3], [p2,p3], [p1,p2,p3] ]
    predictions = {}
    for pixel_set in pixel_sets:
        pixels = tuple(map(get_pixels, pixel_set))
        corrupted = apply_corruptions(data, pixels)
        predictions[pixels] = get_prediction(model, corrupted)
    return predictions
    
def get_prediction(model, data):
    output = model(data)
    weight, pred = torch.max(output,1)
    return int(pred)
    
def apply_corruptions(data, pixels):
    mod_data = data.clone()
    for r,c in pixels:
        mod_data[0,0,r,c] = 1 - data[0,0,r,c]
    return mod_data

def interactions(model=MODEL):
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, 
                   transform=transforms.Compose(models.transform_list())),
    batch_size=1, shuffle=True)

    model.eval()
#    test_loss = 0
#    correct = 0
    image_counter = 0
    for data, target in test_loader:
        image_counter += 1
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        weight, base_pred = torch.max(output, 1)
        print('Image', image_counter, 'Label:', int(target),  'Predicted:', int(base_pred))
        for limit in range(10000):
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
                    
                    
            
#            predictions = dict.fromkeys(range(10), 0)
#            output = model(data)
#            weight, base_pred = torch.max(output, 1)
#            base_pred = int(base_pred) # prediction without any corruptions
            
#            for i in range(28*28):
#                print(i, int(target), base_pred, predictions)
#                for j in range(28*28):
#                    r1 = i % 28; c1 = int((i-r1)/28)
#                    r2 = j % 28; c2 = int((j-r2)/28)
#                    mod_data = data.clone()
#                    mod_data[0,0,r1,c1] = 1 - data[0,0,r1,c2]
#                    mod_data[0,0,r2,c2] = 1 - data[0,0,r2,c2]  
#                    output = model(mod_data)
#                    weight, pred = torch.max(output, 1)
#                    predictions[int(pred)] += 1

#    test_loss /= len(test_loader.dataset)
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))
    
interactions()
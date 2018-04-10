# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 20:52:54 2018

@author: jhyun95
"""

from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models

#from mkl import set_num_threads
#set_num_threads(4)

def main():
    # Training settings   
    CUDA = False; #torch.cuda.is_available() 
    INPUT_BATCH = 64
    TEST_BATCH = 1000
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10
    LOG_INTERVAL = 25  
#    torch.manual_seed(args.seed)  
#    if CUDA:
#        torch.cuda.manual_seed(args.seed)
    
    ''' Download and normalize data set '''
    #kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, 
                       transform=transforms.Compose(models.transform_list())),
        batch_size=INPUT_BATCH, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(models.transform_list())),
        batch_size=TEST_BATCH, shuffle=True, **kwargs)
    
    ''' Model training epochs '''
    model = models.BaseNet()
#    if CUDA:
#        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#                          weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch, LOG_INTERVAL)
        test(model, test_loader)
        
    MODEL_PATH = '../models/BaseNet_E10'
    torch.save(model.state_dict(), MODEL_PATH)
#    model = models.LinNet()
#    model.load_state_dict(torch.load(MODEL_PATH))
        
def train(model, train_loader, optimizer, epoch, log_interval):
    ''' Training step '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
#        if args.CUDA:
#            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, test_loader):
    ''' Testing step '''
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
#        if CUDA.cuda:
#            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()
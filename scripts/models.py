# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 03:13:28 2018

@author: jhyun_000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
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
                       transform=transforms.Compose(transform_list())),
        batch_size=INPUT_BATCH, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(transform_list())),
        batch_size=TEST_BATCH, shuffle=True, **kwargs)
    
    ''' Model training epochs, PICK MODEL TYPE HERE '''
    model = ConvNet() 
#    if CUDA:
#        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)# weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS + 1):
        train_model(model, train_loader, optimizer, epoch, LOG_INTERVAL)
        test_model(model, test_loader)
    torch.save(model.state_dict(), '../models/ConvNet_E10')
#    model = models.BaseNet()
#    model.load_state_dict(torch.load('../models/ConvNet_E10'))
        
def train_model(model, train_loader, optimizer, epoch, log_interval):
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

def test_model(model, test_loader):
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

def transform_default():
    ''' Image scaling, suggested in tutorial. Not used '''
    return [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

def transform_list():
    ''' Grayscale image to black and white. Used for all models '''
    return [transforms.ToTensor(), lambda x: x > 0.5, lambda x: x.float()]
    
class LinearNet(nn.Module):
    ''' 3-layer linear fully connected network '''
    def __init__(self):
        super(LinearNet, self).__init__()
        self.lin1 = nn.Linear(784,112)
        self.lin2 = nn.Linear(112,16)
        self.lin3 = nn.Linear(16,10)
        
    def forward(self, x):
        x = x.view(-1,784)
        x = self.lin1(x); x = F.relu(x)
        x = self.lin2(x); x = F.relu(x)
        x = self.lin3(x); x = F.log_softmax(x)
        return x

class ConvNet(nn.Module):
    ''' Tutorial network for 2 convolutional layers and 2 linear layers '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class StnNet(nn.Module):
    ''' Tutorial Spatial Transformer network, from:
        http://pytorch.org/tutorials/intermediate/
        spatial_transformer_tutorial.html?highlight=mnist '''
    def __init__(self):
        super(StnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    main()
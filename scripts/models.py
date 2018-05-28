# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 03:13:28 2018

@author: jhyun_000
"""

import sys
import numpy as np
import networkx as nx

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
                       transform=transforms.Compose(transform_black_and_white())),
        batch_size=INPUT_BATCH, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose(transform_black_and_white())),
        batch_size=TEST_BATCH, shuffle=True, **kwargs)
    
    ''' Model training epochs, PICK MODEL TYPE HERE '''
    model = ConvNet() 
#    if CUDA:
#        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)# weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS + 1):
        train_model(model, train_loader, optimizer, epoch, LOG_INTERVAL)
        test_model(model, test_loader)
#    torch.save(model.state_dict(), '../models/ConvNet_E10')
#    model = models.ConvNet()
#    model.load_state_dict(torch.load('../models/ConvNet_E10'))
        
def train_model(model, train_loader, optimizer, epoch, log_interval):
    ''' Training step for number predictor model '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
#        if args.CUDA:
#            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_model(model, test_loader):
    ''' Testing step for number predictor model '''
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
#        if CUDA.cuda:
#            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def transform_black_and_white():
    ''' Grayscale image to black and white. Used for all true models '''
    return [transforms.ToTensor(), lambda x: x > 0.5, lambda x: x.float()]

def transform_default():
    ''' Image scaling, suggested in tutorial. Not used '''
    return [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

def make_dcellnet_for_label(true_model, label=2, correlation_data_file=None,
                            p_threshold=0.05, min_cluster_size=10,
                            plot_ontology=True):
    ''' Creates a DCellNet for predicting whether or not an image has a 
        particular label. The intended usage is to be analogous to cell 
        viability predictions from genotype. For instance, taking each pixel 
        in a black and white image as the "genotype", DCellNet will predict 
        whether or not the image's label or "viability" is changed (based
        on true_model), when various pixels are flipped or "knocked out". '''
        
    from interactions import find_pixel_correlations
    from hierarchical_clustering import compute_hierarchical_clusters
    ''' Get pixel correlations for the label subset of images '''
    print('Loading pixel correlations...')
    if correlation_data_file == None: # no precomputed correlations
        correlations = find_pixel_correlations(
                true_model, labelset=label, check_consistency=True,
                mode='mcc-adj', output_file=None)
    else: # precomputed correlation values
        correlations = np.loadtxt(correlation_data_file, delimiter=',')
        
    ''' Construct pixel hierarchy. '''
    print('Computing pixel hierarchical clusters and associations...')
    distances = np.max(correlations) - np.abs(correlations)
    root, adj, assoc, unique_assoc = compute_hierarchical_clusters(
            distances, p_threshold, min_cluster_size, 
            merge_linear_branches=True, plot_dendrogram=False, 
            plot_ontology=plot_ontology)
    
    ''' Format hierarchy for DCellNet object '''
    print('Constructing DCellNet...')
    num_terms, num_elements = assoc.shape
    dG = nx.DiGraph(adj)
    term_size_map = {}; term_element_map = {}
    for i in range(num_terms):
        term = i
        ''' Use non-unique associations for term size, since this only
            dictates the number of neurons assigned to this term '''
        term_size_map[term] = np.sum(assoc[i,:]) 
        ''' Use unique associations for element mapping, to avoid objects
            inputting into every term in the hierarchy (i.e. using non-unique
            associations would have the root neuron set receive num_elements + 
            all children outputs as its input) '''
        directly_mapped_elements = np.nonzero(unique_assoc[i,:])[0] # unique associations
        if len(directly_mapped_elements) > 0:
            term_element_map[term] = set(directly_mapped_elements)
        
    return DCellNet(term_size_map, term_element_map, dG, num_elements, root)


class DCellNet(nn.Module):
    ''' Adapted from jisoo's DCell code, via majianzhu '''

    def __init__(self, term_size_map, term_element_map, dG, num_elements, root):
        ''' Takes ngene inputs as a 1D tensor, returns a single output.
            NOTE: The model will be different based on whether or not elements
            are mapped only once, or if they are mapped to a term as well as
            all of its parent terms, as specificed in term_element_map.
            Mapping only once gives a simpler model.
            
            term_size_map : dict {term label: term size}, exclusively for neuron counts
            term_element_map : dict {term label: set(elements)}, 
            dG: networkx DiGraph, adjacency matrix for hierarchy
            ngene: int, number of elements (not number of terms) 
            root: obj, label of root cluster '''
        super(DCellNet, self).__init__()
        self.root = root
        self.term_element_map = term_element_map # renamed from term_direct_gene_map
        self.feature_dim = num_elements
        self.term_dim_map = {} # number of neurons assigned to each term
        self.term_layer_list = [] # list of layers (one per hierarchy depth, computation order)
        self.term_child_map = {} # term to child terms, renamed from term_neighbor_map
        self.calculate_term_dim(term_size_map)  
        self.construct_direct_input_layers()
        self.construct_NN_graph(dG)

    def calculate_term_dim(self, term_size_map): # renamed from cal_term_dim
        ''' Initializes number of neurons per ontology term, based on term size '''
        for term, term_size in term_size_map.items():
            self.term_dim_map[term] = max( 15, int( 0.3 * term_size))

    def construct_direct_input_layers(self): # renamed from contruct_direct_gene_layer
        ''' Constructs linear input layers for all terms. Reduces full input
            tensor to only the relevant inputs for each node. '''
        for term, elements in self.term_element_map.items():
            if len(elements) == 0:
                print('There are no directed associated elements for term', term)
                sys.exit(1)
            self.add_module(str(term)+'_direct_input_layer', nn.Linear(self.feature_dim, len(elements)))

    def construct_NN_graph(self, dG):
        ''' Constructs linear layers to form neural network hierarchy '''
        for term in dG.nodes():
            self.term_child_map[term] = []
            for child in dG.neighbors(term):
                self.term_child_map[term].append(child)

        leaves = [n for n,d in dG.out_degree() if d==0]
        while len(leaves) > 0:
            self.term_layer_list.append(leaves)
            for term in leaves:
                input_size = 0
                for child in self.term_child_map[term]: # from child terms  
                    input_size += self.term_dim_map[child]
                if term in self.term_element_map: # directly mapped elements
                    input_size += len(self.term_element_map[term])
                term_hidden = self.term_dim_map[term] # num neurons = output dim
                self.add_module(str(term)+'_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(str(term)+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(str(term)+'_aux_linear_layer1', nn.Linear(term_hidden,1))
                self.add_module(str(term)+'_aux_linear_layer2', nn.Linear(1,1))
    
            dG.remove_nodes_from(leaves)
            leaves = [n for n,d in dG.out_degree() if d==0]

    def forward(self, x):
        ''' Forward calculation for training and prediction '''
        term_out_map = {} # renamed from term_gene_out_map
        for term, _ in self.term_element_map.items():
            term_out_map[term] = self._modules[str(term) + '_direct_input_layer'](x) 

        term_NN_out_map = {} # direct tensor output of each term's neuron set
        aux_out_map = {} # processed through additional tanh and linear layers
        for i, layer in enumerate(self.term_layer_list):
            for term in layer: # compute neuron outputs from lowest to highest depth
                child_input_list = []
                for child in self.term_child_map[term]: # input has output from child term neurons
                    child_input_list.append(term_NN_out_map[child])
                if term in self.term_element_map: # input has directly mapped elements
                    child_input_list.append(term_out_map[term])
                child_input = torch.cat(child_input_list,1)
                
                # compute direct output from current term's neurons
                term_NN_out = self._modules[str(term)+'_linear_layer'](child_input)              

                # pass through Tanh and BatchNorm layers before feeding to parent neurons
                Tanh_out = F.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[str(term)+'_batchnorm_layer'](Tanh_out)
                
                # auxillary outputs 
                aux_layer1_out = F.tanh(self._modules[str(term)+'_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[str(term)+'_aux_linear_layer2'](aux_layer1_out)

#        return aux_out_map, term_NN_out_map
        return term_NN_out_map[self.root] # output at the root node
    
    
''' True models for original MNIST problem of predicting image labels '''
    
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
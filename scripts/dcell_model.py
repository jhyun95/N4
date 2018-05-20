# -*- coding: utf-8 -*-
"""
Created on Sun May 20 00:37:09 2018

@author: jhyun_000

Adapted from jisoo's DCell code, via majianzhu
"""

import sys, os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import util
#from util import *

class DCellNet(nn.Module):

    def __init__(self, term_size_map, term_element_map, dG, num_elements, root):
        ''' NOTE: The model will be different based on whether or not elements
            are mapped only once, or if they are mapped to a term as well as
            all of its parent terms. Mapping only once gives a simpler model.
            
            term_size_map : dict {term label: term size} 
            term_element_map : dict {term label: set(elements)}
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
        self.cal_term_dim(term_size_map)  
        self.construct_direct_input_layers()
        self.construct_NN_graph(dG)

    def cal_term_dim(self, term_size_map):
        ''' Initializes number of neurons per ontology term, based on term size '''
        for term, term_size in term_size_map.items():
            self.term_dim_map[term] = max( 15, int( 0.3 * term_size))

    def construct_direct_input_layers(self):
        ''' Constructs linear input layers for all terms. Reduces full input
            tensor to only the relevant inputs for each node.
            Previously called contruct_direct_gene_layer '''
        for term, elements in self.term_element_map.items():
            if len(elements) == 0:
                print('There are no directed associated elements for term', term)
                sys.exit(1)
            self.add_module(term+'_direct_input_layer', nn.Linear(self.feature_dim, len(elements)))

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
                self.add_module(term+'_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden,1))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(1,1))
    
            dG.remove_nodes_from(leaves)
            leaves = [n for n,d in dG.out_degree() if d==0]

    def forward(self, x):
        ''' Forward calculation for training and prediction '''
        term_out_map = {} # renamed from term_gene_out_map
        for term, _ in self.term_element_map.items():
            term_out_map[term] = self._modules[term + '_direct_input_layer'](x) 

        term_NN_out_map = {} # tensor output of each term's neuron set
        aux_out_map = {}
        for i, layer in enumerate(self.term_layer_list):
            for term in layer: # compute neuron outputs from lowest to highest depth
                child_input_list = []
                for child in self.term_child_map[term]: # input has output from child term neurons
                    child_input_list.append(term_NN_out_map[child])
                if term in self.term_element_map: # input has directly mapped elements
                    child_input_list.append(term_out_map[term])
                child_input = torch.cat(child_input_list,1)
                
                # compute direct output from current term's neurons
                term_NN_out = self._modules[term+'_linear_layer'](child_input)              

                # pass through Tanh and BatchNorm layers before feeding to parent neurons
                Tanh_out = F.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
                
                # auxillary outputs 
                aux_layer1_out = F.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)

#        return aux_out_map, term_NN_out_map
        return term_NN_out_map[self.root] # output at the root node
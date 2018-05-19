import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from ontology_NN import *
import argparse

def expand_genotype(genotype, feature_dim):
		
	feature = torch.zeros(genotype.size()[0], feature_dim).float()	
		
	for i in range(genotype.size()[0]):
		feature[i, genotype[i,0]] = 1
		feature[i, genotype[i,1]] = 1				

	return feature

def create_term_mask(term_direct_gene_map, feature_dim):

	term_mask_map = {}

	for term, gene_set in term_direct_gene_map.items():

		mask = torch.zeros(len(gene_set), feature_dim)

		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1

		mask_gpu = mask.cuda()

		term_mask_map[term] = mask_gpu

	return term_mask_map

def train_dcell(root, term_size_map, term_direct_gene_map, dG, train_data, feature_dim, model_save_folder, train_epochs, batch_size, learning_rate):

	loss_map = {}
	for term, _ in term_size_map.items():
		loss_map[term] = nn.MSELoss()

	model = dcell_nn(term_size_map, term_direct_gene_map, dG, feature_dim, root)

	train_feature, train_label, test_feature, test_label = train_data

	train_label_gpu = train_label.cuda()
	test_label_gpu = test_label.cuda()

	model.cuda()
	term_mask_map = create_term_mask(model.term_direct_gene_map, feature_dim)

	for name, param in model.named_parameters():
		term_name = name.split('_')[0]
		if '_direct_gene_layer.weight' in name:
			#print name, param.size(), term_mask_map[term_name].size()
			param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
		else:
			param.data = param.data * 0.1

	train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
	test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)

	for epoch in range(train_epochs):

		#Train
		model.train()
		train_predict = torch.zeros(0,0).cuda()
			
		for i, (genotypes, labels) in enumerate(train_loader):
			# Convert torch tensor to Variable
			features = expand_genotype(genotypes, feature_dim)

			cuda_features = Variable(features.cuda())
			cuda_labels = Variable(labels.cuda())

			# Forward + Backward + Optimize
			optimizer.zero_grad()  # zero the gradient buffer
			aux_out_map,_ = model(cuda_features)

			train_predict = torch.cat([train_predict, aux_out_map[root].data],0)

			total_loss = 0
			for term, loss in loss_map.items():
				outputs = aux_out_map[term]
				if term == root:	
					total_loss += loss_map[term](outputs, cuda_labels)
				else:
					total_loss += 0.2 * loss_map[term](outputs, cuda_labels)

			total_loss.backward()

			for name, param in model.named_parameters():
				if '_direct_gene_layer.weight' not in name:
					continue
				term_name = name.split('_')[0]
				#print name, param.grad.data.size(), term_mask_map[term_name].size()
				param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

			optimizer.step()

		train_corr = spearman_corr(train_predict, train_label_gpu)

		if epoch % 10 == 0:
			torch.save(model, model_save_folder + '/model_' + str(epoch))

		#Test
		model.eval()
		
		test_predict = torch.zeros(0,0).cuda()
		for i, (genotypes, labels) in enumerate(test_loader):
			# Convert torch tensor to Variable
			features = expand_genotype(genotypes, feature_dim)
			cuda_features = Variable(features.cuda())

			aux_out_map,_ = model(cuda_features)
			test_predict = torch.cat([test_predict, aux_out_map[root].data],0)

		test_corr = spearman_corr(test_predict, test_label_gpu)

		if epoch % 10 == 0:
			print 'Epoch', epoch, 'train corr', train_corr, 'test corr', test_corr


	torch.save(model, model_save_folder + '/model_final')	
	#model.cpu()
	model2 = torch.load('MODEL/model_final')
	model2.eval()

	features = expand_genotype(test_feature, feature_dim)
	cuda_features = Variable(features.cuda())
	aux_out_map = model2(cuda_features)
	test_corr = spearman_corr(aux_out_map[root].data, test_label_gpu)	
	print 'reload model corr', test_corr

parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-train', help='Training dataset', type=str)
parser.add_argument('-test', help='Validation dataset', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=500)
parser.add_argument('-model', help='Folder for trained models', type=str, default='MODEL/')
opt = parser.parse_args()

torch.set_printoptions(precision=5)

train_data, gene2id_mapping = prepare_train_data(opt.train, opt.test)

dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

save_gene2id_mapping(gene2id_mapping, root, opt.model)

train_dcell(root, term_size_map, term_direct_gene_map, dG, train_data, len(gene2id_mapping), opt.model, opt.epoch, opt.batchsize, opt.lr)	

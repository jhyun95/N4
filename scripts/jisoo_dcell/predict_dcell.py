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

def predict_dcell(predict_data, feature_dim, model_file, hidden_folder, batch_size, result_file):

	model = torch.load(model_file)

	predict_feature, predict_label = predict_data

	predict_label_gpu = predict_label.cuda()

	model.cuda()
	model.eval()

	test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)

	#Test
	test_predict = torch.zeros(0,0).cuda()
	
	batch_num = 0
	for i, (genotypes, labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		features = expand_genotype(genotypes, feature_dim)
		cuda_features = Variable(features.cuda(),volatile=True, requires_grad=False)

		aux_out_map, term_hidden_map = model(cuda_features)
		test_predict = torch.cat([test_predict, aux_out_map[model.root].data],0)

		for term, hidden_map in term_hidden_map.items():
			np.savetxt(hidden_folder+'/'+term+'_'+str(i)+'.txt', hidden_map.data.cpu().numpy(), '%.4e')	

		batch_num += 1

	test_corr = spearman_corr(test_predict, predict_label_gpu)
	print 'Test pearson corr', model.root, test_corr	

	for term, _ in model.term_dim_map.items():
		hidden_file = hidden_folder+'/'+term+'.hidden'
		for i in range(batch_num):
	 		os.system('cat ' + hidden_folder+'/'+term+'_'+str(i)+'.txt >> ' + hidden_file)
			os.system('rm ' + hidden_folder+'/'+term+'_'+str(i)+'.txt')

	np.savetxt(result_file+'/'+model.root+'.predict', test_predict.cpu().numpy(),'%.4e')		
	#for term, hidden in model._modules.items():
	#	print term, type(hidden)

parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-predict', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default=1000)
parser.add_argument('-load', help='Model file', type=str, default='MODEL/model_200')
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='Hidden/')
parser.add_argument('-result', help='Result file name', type=str, default='Result/')
opt = parser.parse_args()

torch.set_printoptions(precision=5)

gene2id_mapping = load_gene2id_mapping(opt.gene2id)
predict_data = prepare_predict_data(testing_file, gene2id_mapping)

predict_dcell(predict_data, len(gene2id_mapping), opt.load, opt.hidden, opt.batchsize, opt.result)	

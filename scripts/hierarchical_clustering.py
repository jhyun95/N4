''' From pyNexo_Fast.py, by majianzhu '''

import numpy as np
import time
from scipy.cluster import hierarchy
from scipy.stats import mannwhitneyu

def hierarchial_clustering(D, threshold, min_size, ncpus):
	#Here D should be a numpy matrix on CPU
	ngene, _ = D.shape
	time1 = time.time()
	
	Z = hierarchy.linkage( D[np.triu_indices(ngene,1)], 'average' )
	Z = Z.astype(int)
	
	time2 = time.time()
	print('Hierarchical clustering takes', time2 - time1,'seconds')

	nmerge = Z.shape[0]
	nterm = nmerge + ngene
	
	global adj
	adj = np.zeros((nterm, nterm))
	global assoc
	assoc = np.zeros((nterm, ngene))
	assoc[0:ngene, 0:ngene] = np.eye(ngene)
	pvalue_list = np.zeros((nterm,))

	for i in range(nmerge):

		C = [ assoc[ Z[i,0], :], assoc[ Z[i,1], :] ]

		adj[ ngene+i, Z[i,0] ] = 1 # cluster ngene+i has two children Z[i,0] and Z[i,1]
		adj[ ngene+i, Z[i,1] ] = 1
		assoc[ ngene+i, :] = np.logical_or(C[0] , C[1]) # genes of ngene+i = genes of Z[i,0] + genes of Z[i,1]
		
		for child in range(2):
			if Z[i,child] >= ngene:
				
				k = Z[i,child] - ngene  #parent term k index

				A1 = assoc[ Z[k,0], :].nonzero()[0] #child1 of term k
				B = assoc[ Z[k,1], :].nonzero()[0]  #child2 of term k
				CC = C[1-child].nonzero()[0]

				A_B_interaction = D[A1,:][:,B].flatten()
				A_C_interaction = D[A1,:][:,CC].flatten()
				B_C_interaction = D[B,:][:,CC].flatten()

				AC_BC_interaction = np.concatenate((A_C_interaction, B_C_interaction), axis=0)
				pvalue_list[ Z[i,child] ] = mannwhitneyu(A_B_interaction, AC_BC_interaction, alternative='less')[1]

	time3 = time.time()
	print('Calculate pvalue takes', time3 - time2,'seconds')

	to_remove = pvalue_list > threshold
	keep = pvalue_list <= threshold
	nterm = np.sum(keep)

	assoc = assoc[keep, :]

	for i in to_remove.nonzero()[0]:
		for p in (adj[:,i]==1).nonzero()[0]:
			adj[p,:] = np.logical_or(adj[p,:], adj[i,:])

		adj[i,:] = 0
		adj[:,i] = 0

	adj = adj[keep,:][:,keep]

	filt = np.sum(assoc, axis=1) >= min_size

	assoc = assoc[filt,:]
	adj = adj[filt,:][:,filt]
	nterm = assoc.shape[0]
	
	time4 = time.time()
	
	print('Prune ontology takes', time4 - time3,'seconds')
	print(adj.shape)
	print(assoc.shape)
	#return adj, assoc, Z

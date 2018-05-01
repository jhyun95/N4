''' Adapted from pyNexo_Fast.py, by majianzhu '''

import time
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import networkx as nx
# For viz, needs pygraphviz
# https://stackoverflow.com/questions/40528048/pip-install-pygraphviz-no-package-libcgraph-found

def hierarchial_clustering(D, p_threshold, min_cluster_size, 
                           plot_dendrogram=False,
                           plot_ontology=False):
    ngene, _ = D.shape
    time1 = time.time()
    
    ''' Hierarchical clustering via SciPy '''
    condensed = ssd.squareform(D)
    Z = sch.linkage(condensed, 'average')
    if plot_dendrogram:
        fig1, ax1 = plt.subplots()
        sch.dendrogram(Z, ax=ax1, no_labels=True)
    Z = Z.astype(int)
    time2 = time.time()
    print('Hierarchical clustering takes', time2 - time1,'seconds')

    ''' Merge clusters based on p-values from Mann-Whitney U test '''
    nmerge = Z.shape[0]
    nterm = nmerge + ngene
    adj = np.zeros((nterm, nterm))
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
                
                ''' Edit: Address elements that are all identical, which is 
                    not handled by native mannwhitneyu, i.e. testing 
                    [1,1] vs [1,1,1]. Assign pvalue = 1.0 for these cases '''
                try:
                    pvalue = mannwhitneyu(A_B_interaction, AC_BC_interaction, alternative='less')[1]
                except ValueError:
                    pvalue = 1.0
                pvalue_list[ Z[i,child] ] = pvalue

    time3 = time.time()
    print('Calculate pvalue takes', time3 - time2,'seconds')

    ''' Prune ontology based on p-value threshold '''
    to_remove = pvalue_list > p_threshold
    keep = pvalue_list <= p_threshold
    nterm = np.sum(keep)
    assoc = assoc[keep, :]
    for i in to_remove.nonzero()[0]:
        for p in (adj[:,i]==1).nonzero()[0]:
            adj[p,:] = np.logical_or(adj[p,:], adj[i,:])
        adj[i,:] = 0
        adj[:,i] = 0

    adj = adj[keep,:][:,keep]
    filt = np.sum(assoc, axis=1) >= min_cluster_size
    assoc = assoc[filt,:]
    adj = adj[filt,:][:,filt]
    nterm = assoc.shape[0]
    time4 = time.time()
    print('Prune ontology takes', time4 - time3,'seconds')
    print('Cluster adjacencies:', adj.shape)
    print('Cluster associations:', assoc.shape)
    
    ''' Visualize ontology '''
    if plot_ontology:
        nclust, _ = adj.shape
        G = nx.from_numpy_matrix(adj)
        clust_sizes = np.sum(assoc, 1)
        labels = {}
        for i in range(nclust):
            labels[i] = str(int(clust_sizes[i]))
        try: # if pygraphviz is available
            layout = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            nx.draw(G, pos=layout, arrows=True, node_size=400, 
                    with_labels=True, labels=labels)    
        except ImportError:
            print('No pygraphviz, using spring layout')
            nx.draw_spring(G, arrows=True, node_size=400, with_labels=True, labels=labels)    
        
    return adj, assoc, Z

data_file = '../data/pixel_correlations/pixel_mcc_adj_3s.csv.gz'
corr = np.loadtxt(data_file, delimiter=',')
distances = np.max(corr) - np.abs(corr)

adj, assoc, Z = hierarchial_clustering(distances, 0.05, 2, False, True)
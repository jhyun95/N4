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
# pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" 
# --install-option="--library-path=/usr/lib/graphviz/"

def main():
    ''' Applies hierarchical clustering of pixels based on pairwise 
        adjusted MCCs between pixels for all images labeled as 3. '''
    data_file = '../data/pixel_correlations/pixel_mcc_adj_3s.csv.gz'
    corr = np.loadtxt(data_file, delimiter=',')
    distances = np.max(corr) - np.abs(corr)
    root, adj, assoc, unique_assoc = compute_hierarchial_clusters( 
            distances, 0.05, 10, True, False, True)
    
def merge_branches(adj, assoc):
    ''' Merges long linear branches into single nodes '''
    
    n0 = adj.shape[0]
    ''' Compute which groups of nodes lie on linear branches and should be merged '''
    merge_groups = []
    for node in range(n0):
        parent = __get_parent__(node, adj)
        if parent != None:
            if len(__get_children__(parent, adj)) == 1: 
                ''' If parent has only one child, merge '''
                merged = False
                for group in merge_groups:
                    if node in group or parent in group: # merge to existing group
                        group.add(node); group.add(parent)
                        merged = True; break
                if not merged:
                    merge_groups.append(set([node, parent]))
    
    ''' Using merge groupings, re-index to preserve original node ordering '''
    reindex = {}; all_merged = set()
    removed_indices = []
    for group in merge_groups: # map groups to the earliest index in group
        remap = min(group) 
        for node in group:
            reindex[node] = remap
            all_merged.add(node)
    for node in range(n0): # retain unmapped indices
        if not node in all_merged:
            reindex[node] = node
        elif not node in reindex.values():
            removed_indices.append(node)
            
    ''' Create adjacency matrix with merged nodes '''
    new_adj = np.zeros((n0,n0))
    for i in range(n0): # reindex original adjacency matrix
        for j in range(n0):
            new_adj[ reindex[i], reindex[j] ] = np.logical_or( \
                new_adj[ reindex[i], reindex[j] ], adj[i,j])
    for i in reversed(removed_indices): # remove unused indices
        new_adj = np.delete(new_adj, i, axis=0)
        new_adj = np.delete(new_adj, i, axis=1)
    for i in range(new_adj.shape[0]): # remove self edges
        new_adj[i,i] = 0 
    
    ''' Update associations '''
    new_assoc = np.zeros(np.shape(assoc))
    for i in range(n0):
        new_assoc[reindex[i], :] = np.logical_or( \
            new_assoc[reindex[i], :], assoc[i,:])
    for i in reversed(removed_indices):
        new_assoc = np.delete(new_assoc, i, axis=0)
    return new_adj, new_assoc    

def compute_hierarchial_clusters(D, p_threshold, min_cluster_size, 
                                 merge_linear_branches=True,
                                 plot_dendrogram=False, plot_ontology=False):
    ''' Generates hierarchical clusters from pairwise distances based on a 
        p-value threshold and minimum cluster size, then merges along
        Returns four outputs: the index of the root cluster, adjacency matrix 
        for clusters, binary association  matrix (cluster x element), 
        and unique binary association matrix (cluster x element, but 
        associates each element to the lowest depth cluster only) '''
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
    print('No Branch Merging:')
    print('    Cluster adjacencies:', adj.shape)
    print('    Cluster associations:', assoc.shape)
    depth = __get_hierarchy_depth__(adj, assoc)
    print('    Hierarchy depth:', depth)
    if merge_linear_branches: # merge linear branches into single node
        adj_merged, assoc_merged = merge_branches(adj, assoc)
        print('After Branch Merging:')
        print('    Cluster adjacencies:', adj_merged.shape)
        print('    Cluster associations:', assoc_merged.shape)
        depth = __get_hierarchy_depth__(adj_merged, assoc_merged)
        print('    Hierarchy depth:', depth)
        adj = adj_merged; assoc = assoc_merged
    unique_assoc = cleanup_multiple_assignments(adj, assoc)
    
    ''' Visualize ontology '''
    if plot_ontology:
        ''' Draw graph hierarchy '''
        plt.figure()
        nclust, _ = adj.shape
        G = nx.from_numpy_matrix(adj)
        clust_sizes = np.sum(assoc, 1)
        unique_clust_sizes = np.sum(unique_assoc, 1)
        child_clust_sizes = clust_sizes - unique_clust_sizes
        labels = {}
        for i in range(nclust):
            labels[i] = str(i) + ":\n" + str(int(child_clust_sizes[i])) \
                + ',' + str(int(unique_clust_sizes[i]))
        try: # if pygraphviz is available
            layout = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            nx.draw(G, pos=layout, arrows=True, node_size=400, 
                    with_labels=True, labels=labels, font_size=8)    
        except ImportError:
            print('No pygraphviz, using spring layout')
            nx.draw_spring(G, iterations=400, arrows=True, node_size=400, 
                           with_labels=True, labels=labels, font_size=8)
            
    root = __get_root__(adj, assoc)
    return root, adj, assoc, unique_assoc

def cleanup_multiple_assignments(adj, assoc):
    ''' Cleans up association matrix so that every element is assigned
        to only one cluster, the lowest depth cluster. Sparser association
        allows for a simpler hierarchical neural network. '''
    num_clusters, num_obj = assoc.shape
    root = __get_root__(adj, assoc)
    new_assoc = np.copy(assoc)
    unvisited = [root]
    while len(unvisited) > 0:
        node = unvisited.pop()
        children = __get_children__(node, adj, True)
        children_assoc = np.zeros(num_obj)
        for child in children: # find all objects assigned to children
            children_assoc = np.logical_or(children_assoc, assoc[child])
        new_assoc[node] -= children_assoc # remove child-overlapping associations
        unvisited += children
    return new_assoc

def __get_parent__(node, adj):
    ''' Get the parent of a node in a tree from an adjacency matrix '''
    parent = np.nonzero(adj[:,node])[0]
    parent = parent[0] if len(parent) > 0 else None
    return parent

def __get_children__(node, adj, as_list=False):
    ''' Get the children of a node in a tree from an adjacency matrix '''
    children = np.nonzero(adj[node,:])[0]
    if as_list: # convert from numpy array to python list
        children = children.tolist()
    return children

def __get_root__(adj, assoc):
    ''' Finds the index of the root in the hierarchy, based on which
        cluster has all objects associated with it. '''
    num_clusters, num_obj = assoc.shape
    clust_sizes = np.sum(assoc, 1)
    for i in range(num_clusters):
        if clust_sizes[i] == num_obj:
            return i

def __get_hierarchy_depth__(adj, assoc):
    ''' Finds the depth of the hierarhcy '''
    root = __get_root__(adj, assoc)
    node_depths = {root:0}
    unvisited = __get_children__(root, adj, True)
    while len(unvisited) > 0:
        node = unvisited.pop()
        if not node in node_depths: # unvisited
            depth = node_depths[__get_parent__(node, adj)] + 1
            node_depths[node] = depth
            unvisited += __get_children__(node, adj, True)
    return max(node_depths.values())
    
if __name__ == '__main__':
    main()
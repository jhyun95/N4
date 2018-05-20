import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

def spearman_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)
    
    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))
        

def load_ontology(file_name, gene2id_mapping):

    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}

    file_handle = open(file_name)

    for line in file_handle:

        line = line.rstrip().split()
        
        if line[2] == 'default':
            dG.add_edge(line[0], line[1])
        else:
            if line[1] not in gene2id_mapping:
                continue

            if line[0] not in term_direct_gene_map:
                term_direct_gene_map[ line[0] ] = set()

            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

    file_handle.close()

    for term in dG.nodes():
        
        term_gene_set = set()

        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]

        deslist = nxadag.descendants(dG, term)

        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        if len(term_gene_set) == 0:
            print 'There is empty terms, please delete term:', term
            sys.exit(1)

        term_size_map[term] = len(term_gene_set)

    #leaves = [n for n,d in dG.in_degree().items() if d==0]
    leaves = [n for n,d in dG.in_degree() if d==0]

    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print 'There are', len(leaves), 'roots:', leaves[0]
    print 'There are', len(dG.nodes()), 'terms'
    print 'There are', len(connected_subG_list), 'connected componenets'

    if len(leaves) > 1:
        print 'There are more than 1 root of ontology. Please use only one root.'
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print 'There are more than connected components. Please connect them.'
        sys.exit(1)

    return dG, leaves[0], term_size_map, term_direct_gene_map

def load_train_data(file_name):

    train_info_list = []
    gene_set = set()

    file_handle = open(file_name)

    for line in file_handle:

        line = line.rstrip().split()
        gene_num = int(line[0])     

        info_list = []
        for i in range(gene_num):
            gene_set.add(line[i+1])
            info_list.append(line[i+1])

        info_list.append(float(line[gene_num+1]))

        train_info_list.append(info_list)

    file_handle.close()

    return train_info_list, gene_set

def transform_data(train_info_list, gene2id_mapping):

    train_feature = torch.zeros(len(train_info_list), 2).int()
    train_label = torch.zeros(len(train_info_list), 1).float()

    for i in range(len(train_info_list)):
        if len(train_info_list[i]) == 3:
            gene1, gene2, GI = train_info_list[i]
            
            train_feature[i, 0] = gene2id_mapping[gene1]
            train_feature[i, 1] = gene2id_mapping[gene2]
            train_label[i, 0] = GI
        else:
            #print train_info_list[i]
            gene, GI = train_info_list[i]
            train_feature[i,0] = gene2id_mapping[gene]
            train_label[i, 0] = GI

    return train_feature, train_label

def prepare_predict_data(test_file, gene2id_mapping):

    test_info_list, test_gene_set = load_train_data(test_file)

    print 'Total gene numbers', len(gene2id_mapping)

    test_feature, test_label = transform_data(test_info_list, gene2id_mapping)

    return (test_feature, test_label)

def load_gene2id_mapping(gene2id_mapping_file):

    gene2id_mapping = {}

    file_handle = open(gene2id_mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        gene2id_mapping[line[0]] = int(line[1])

    file_handle.close()
    
    return gene2id_mapping

def save_gene2id_mapping(gene2id_mapping, root, model_save_folder):
    
    file_handle = open(model_save_folder + '/' + root + '_gene2id_mapping', 'w')

    for gene_name, gene_id in gene2id_mapping.items():
        file_handle.writelines(gene_name + ' ' + str(gene_id) + '\n')

    file_handle.close()

def prepare_train_data_server(train_file, test_file, gene2id_mapping):

    train_info_list, train_gene_set = load_train_data(train_file)
    test_info_list, test_gene_set = load_train_data(test_file)

    gene_set = train_gene_set | test_gene_set

    print 'Total gene numbers', len(gene_set)

    train_feature, train_label = transform_data(train_info_list, gene2id_mapping)
    test_feature, test_label = transform_data(test_info_list, gene2id_mapping)

    return (train_feature, train_label, test_feature, test_label), gene2id_mapping

def prepare_train_data(train_file, test_file):

    train_info_list, train_gene_set = load_train_data(train_file)
    test_info_list, test_gene_set = load_train_data(test_file)
    
    gene_set = train_gene_set | test_gene_set

    print 'Total gene numbers', len(gene_set)

    gene2id_mapping = dict(zip(list(gene_set), range(len(gene_set))))

    train_feature, train_label = transform_data(train_info_list, gene2id_mapping)
    test_feature, test_label = transform_data(test_info_list, gene2id_mapping)

    return (train_feature, train_label, test_feature, test_label), gene2id_mapping

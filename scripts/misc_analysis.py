# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:16:59 2018

@author: jhyun_000
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ['DCell_A', 'DCell_B', 'DCell_C', 'DCell_D', 'DCell_E']

def main():
    ''' Plot % positive vs % negative pairwise interactions in all images with essential px '''
#    plot_pairwise_interaction()
    
    ''' Plot DCell performance plots (training and evaluation) '''
#    df_eval, df_fit = parse_perf_log(log_file='../data/logs/log_min20_epochs100_extended.txt')
#    plot_evaluation_performance(df_eval)
#    plot_training_performance(df_fit)
    
    ''' Plot DCell vs Fully connected (evaluation only) '''
    
#    df_eval_fc, df_fit_fc = parse_perf_log(log_file='../data/logs/log_min20_epochs100_fc_extended.txt')
#    print(df_eval.head())
#    plot_evaluation_performance(df_eval)
    
    ''' Plot % lethal vs KO size '''
#    plot_lethal_ko_percents(limit=None)
    
    ''' Plot % positive/negative interactions vs KO size'''
#    plot_interactions()
    
def plot_interactions(interactions_file='../data/logs/log_interactions.txt',
                      single_plot=False):
    ''' Plots percent of KOs with positive or negative interactions '''
    KO_LIMIT = 10
    PAIR_POSITIVE = [1.51, 2.49, 2.54, 2.32, 2.74] # double KO positive interactions
    PAIR_NEGATIVE = [2.98, 3.30, 2.52, 2.52, 3.19] # double KO negative interactions

    raw_values = []
    for line in open(interactions_file, 'r+'):
        if '>' == line[0]:
            rate = line.split()[-1]
            rate = float(rate[1:-2])
            raw_values.append(rate)
    n = len(raw_values); m = len(MODELS)
    raw_values = np.array(raw_values)
    raw_values = np.reshape(raw_values, (int(n/3),3)) 
    raw_values = np.reshape(raw_values, (m,int(n/3/m),3))
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,4))
    ko_range = np.arange(2, KO_LIMIT+1)
    for i in range(m):
        positive_interactions = [PAIR_POSITIVE[i]] + raw_values[i,:,1].tolist()
        negative_interactions = [PAIR_NEGATIVE[i]] + raw_values[i,:,2].tolist()
        axs[0].plot(ko_range, positive_interactions, label=MODELS[i])
        axs[1].plot(ko_range, negative_interactions, label=MODELS[i])
    axs[0].set_xlabel('KO size'); axs[0].set_ylabel('% Positive Interaction')
    axs[1].set_xlabel('KO size'); axs[1].set_ylabel('% Negative Interaction')
    axs[0].set_title('Positive Interactions')
    axs[1].set_title('Negative Interactions')
    axs[1].legend()
    plt.tight_layout()
            
    
def plot_lethal_ko_percents(lethal_count_file='../data/logs/log_lethal_200.txt',
                            lethal_count_file2='../data/logs/log_lethal_201-500.txt',
                            limit=None):
    ''' Plots the percent of KOs being lethal vs. KO size. Combines data
        from two logs. '''
    KO_START = 2; KO_LIMIT = 200; KO_LIMIT2 = 500
    SINGLE_KO = np.array([35, 73, 51, 36, 53]) / 784 * 100 # single KO lethal cases
        
    ''' Extract from first file '''
    lethal_rates = np.zeros((len(MODELS), KO_LIMIT2+1))
    count = KO_START; model = 0
    for line in open(lethal_count_file, 'r+'):
        if '>' == line[0]:
            lethal_rate = line.split()[-1]
            lethal_rate = float(lethal_rate[1:-2])
            lethal_rates[model, count] = lethal_rate
            count += 1
            if count > KO_LIMIT:
                count = KO_START; model += 1
    
    ''' Extract from second file '''     
    count = KO_LIMIT + 1; model = 0
    for line in open(lethal_count_file2, 'r+'):
        if '>' == line[0]:
            lethal_rate = line.split()[-1]
            lethal_rate = float(lethal_rate[1:-2])
            lethal_rates[model, count] = lethal_rate
            count += 1
            if count > KO_LIMIT2:
                count = KO_LIMIT + 1; model += 1
    
    ''' Generate plot with concatenated data '''           
    lethal_rates[:,1] = SINGLE_KO
    ko_counts = np.arange(0,KO_LIMIT2+1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4))
    limit = limit if limit != None else len(ko_counts+1)
    for i in range(len(MODELS)):
        ax.plot(ko_counts[:limit], lethal_rates[i,:limit], label=MODELS[i].replace('DCell','ConvNet'))
    ax.set_title('% Lethal vs KO size')
    ax.legend()
    ax.set_ylabel('% Lethal')
    ax.set_xlabel('KO size')
    plt.tight_layout()
    
def plot_evaluation_performance_comparison(df_eval, df_eval_fc):
    ''' Plots the MCC uring evaluation against different sized KOs 
        for DCell-like models vs fully connected DCell models '''
    df = df_eval

    
def plot_evaluation_performance(df_eval, start_ko=2, end_ko=15):
    ''' Plots the MCC and ACC during evaluation against different sized KOs '''
    df = df_eval
    rows, cols = df.shape
    KO_counts = range(start_ko, end_ko+1)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3))
    for m in range(len(MODELS)):
        mcc = []; acc = []; label = MODELS[m]
        row_ind = len(KO_counts) * m
        mcc = df.loc[row_ind:row_ind+len(KO_counts)-1, 'MCC']
        acc = df.loc[row_ind:row_ind+len(KO_counts)-1, 'ACC']
        axs[0].plot(KO_counts, mcc, label=label)
        axs[1].plot(KO_counts, acc, label=label)
    axs[0].set_title('MCC versus KO size')
    axs[1].set_title('Accuracy versus KO size')
    axs[0].legend()
    axs[0].set_ylabel('MCC')
    axs[1].set_ylabel('Accuracy')
    axs[0].set_xlabel('KO size')
    axs[1].set_xlabel('KO size')
    plt.tight_layout()
    
def plot_training_performance(df_fit, epochs=100):
    ''' Plots the MCC in the training and testing sets vs. epoch '''
    df = df_fit
    print(df.head())
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3), sharey=True)
    x = range(epochs + 1)
    for i in range(len(MODELS)):
        training_mcc = df.values[:,i+len(MODELS)+1]
        testing_mcc = df.values[:,i+1]
        label = MODELS[i]
        axs[0].plot(x, training_mcc, label=label)
        axs[1].plot(x, testing_mcc, label=label)
    axs[0].set_title('MCC for training set')
    axs[1].set_title('MCC for testing set')
    axs[0].legend()
    axs[0].set_ylabel('MCC')
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    plt.tight_layout()
    
def parse_perf_log(log_file='../data/logs/log_min20_epochs100_extended.txt', 
                   epochs=100, start_ko=2, end_ko=15):
    ''' Extract ACC and MCC curves for each model '''
    raw_values = []
    for line in open(log_file, 'r+'):
        if line[0] == '>':
            value = line.split(':')[-1].strip()
            value = float(value) if '.' in value else int(value)
            raw_values.append(value)
    KO_counts = range(start_ko, end_ko+1)
    df_eval = pd.DataFrame(columns=['model', 'KO_order', 
        'MCC', 'ACC', 'TP', 'FP', 'FN', 'TN'])
    row_num = 0; model_ID = 0; KO_ID = 0
    for i in range(0,len(raw_values),6):
        model = MODELS[model_ID]
        KOcount = KO_counts[KO_ID]
        df_eval.loc[row_num] = [model, KOcount] + raw_values[i:i+6] 
        row_num += 1; KO_ID += 1
        if KO_ID >= len(KO_counts):
            KO_ID = 0; model_ID += 1
            
    ''' Extract testing and training MCC curves vs epoch '''
    training = []; testing = []
    for line in open(log_file, 'r+'):
        if 'Training MCC' in line:
            training.append( float(line.split()[2]) )
        elif 'Testing MCC' in line:
            testing.append( float(line.split()[2]) )
    df_fit = pd.DataFrame(columns=['epoch', 'A_test_mcc', 
                       'B_test_mcc', 'C_test_mcc', 
                       'D_test_mcc', 'E_test_mcc',
                       'A_train_mcc', 'B_train_mcc',
                       'C_train_mcc', 'D_train_mcc',
                       'E_train_mcc'])
    df_fit.loc[0] = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range(epochs):
        training_mccs = map(lambda x: training[i+epochs*x], range(len(MODELS)))
        testing_mccs = map(lambda x: testing[i+epochs*x], range(len(MODELS)))
        row = [i+1] + list(testing_mccs) + list(training_mccs)
        df_fit.loc[i+1] = row
    
    return df_eval, df_fit
  
    
def plot_pairwise_interaction():
    DIM = 28
    FIRST_ORDER_FILE = '../data/DCell_test/1st_order.tsv'
    SECOND_ORDER_FILE = '../data/DCell_test/2nd_order.tsv'
    REAL_PINT = 1.9; REAL_NINT = 3.1
    num_pixels = DIM*DIM
    num_pairs = num_pixels * (num_pixels-1) / 2
    df_1st = pd.read_csv(FIRST_ORDER_FILE, sep='\t', index_col=0)
    lethal_counts = []
    for image in df_1st.index:
        num_lethal = len(df_1st.loc[image,'corrupting_pixels'].split(';'))
        lethal_counts.append(num_lethal)
    df_1st['num_lethal'] = lethal_counts
    
    df_2nd = pd.read_csv(SECOND_ORDER_FILE, sep='\t', index_col=0)
    rows, cols = df_2nd.shape
    df_2nd['num_lethal'] = np.zeros(rows, dtype=np.int)
    for image in df_2nd.index:
        df_2nd.loc[image, 'num_lethal'] = df_1st['num_lethal'][image]
    df_2nd['frac_lethal'] = df_2nd['num_lethal'] / num_pixels
        
    pos_ints = df_2nd['positive_interactions'].values / num_pairs * 100
    neg_ints = df_2nd['negative_interactions'].values / num_pairs * 100
    dist = np.square(pos_ints - REAL_PINT) + np.square(neg_ints - REAL_NINT)
    df_2nd['difference'] = dist
    df_2nd.sort_values('difference', inplace=True)
    print(df_2nd.head())
    
    ''' Separate histograms for interaction counts '''
#    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3))
#    bins = np.arange(0,10.5,0.5)
#    yticks = np.arange(0, 21, 5)
#    axs[0].hist(pos_ints, bins=bins)
#    axs[1].hist(neg_ints, bins=bins)
#    axs[0].set_ylim([0,20]); axs[0].set_yticks(yticks)
#    axs[1].set_ylim([0,20]); axs[1].set_yticks(yticks)
    
    ''' Single scatterplot for interaction counts '''
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.scatter(pos_ints, neg_ints, label='images')
    ax.scatter([REAL_PINT], [REAL_NINT], label='yeast')
    ax.set_xlabel('% of pairs with positive interactions')
    ax.set_ylabel('% of pairs with negative interactions')
    plt.legend(loc='upper right')
    
if __name__ == '__main__':
    main()
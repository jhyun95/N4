# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:16:59 2018

@author: jhyun_000
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ['DCell_A', 'DCell_B', 'DCell_C', 'DCell_D', 'DCell_E']
EPOCHS_IN_LOG = 100

def main():
#    plot_pairwise_interaction()
#    parse_log(log_file='/mnt/346490BF64908570/log_min20_epochs5.txt',
#              out_fit_file='../fitting.csv')
    parse_log()
    plot_training_performance()
    plot_evaluation_performance()
    
def plot_evaluation_performance(eval_file='../data/DCell_test/evaluation.csv'):
    ''' Plots the MCC and ACC during evaluation against different sized KOs '''
    df = pd.read_csv(eval_file)
    rows, cols = df.shape
    KO_counts = range(2,15+1)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3))
    for m in range(len(MODELS)):
        mcc = []; acc = []; label = MODELS[m]
        for i in range(len(KO_counts)):
            row_ind = len(KO_counts) * m + i
            mcc.append(df.loc[row_ind, 'MCC'])
            acc.append(df.loc[row_ind, 'ACC'])
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
    
def plot_training_performance(perf_file='../data/DCell_test/fitting.csv'):
    ''' Plots the MCC in the training and testing sets vs. epoch '''
    df = pd.read_csv(perf_file, index_col=0)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3), sharey=True)
    x = range(EPOCHS_IN_LOG + 1)
    for i in range(len(MODELS)):
        training_mcc = df.values[:,i+len(MODELS)]
        testing_mcc = df.values[:,i]
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
    
def parse_log(log_file='../data/DCell_test/log_min20_epochs100_extended.txt',
              out_eval_file='../data/DCell_test/evaluation.csv',
              out_fit_file='../data/DCell_test/fitting.csv'):
    ''' Extract ACC and MCC curves for each model '''
    raw_values = []
    for line in open(log_file, 'r+'):
        if line[0] == '>':
            value = line.split(':')[-1].strip()
            value = float(value) if '.' in value else int(value)
            raw_values.append(value)
    print(raw_values)
    max_KO = 15
    KO_counts = range(2,max_KO+1)
    df = pd.DataFrame(columns=['model', 'KO_order', 
        'MCC', 'ACC', 'TP', 'FP', 'FN', 'TN'])
    row_num = 0; model_ID = 0; KO_ID = 0
    for i in range(0,len(raw_values),6):
        model = MODELS[model_ID]
        KOcount = KO_counts[KO_ID]
        df.loc[row_num] = [model, KOcount] + raw_values[i:i+6] 
        row_num += 1; KO_ID += 1
        if KO_ID >= len(KO_counts):
            KO_ID = 0; model_ID += 1
    df.to_csv(out_eval_file, sep=',', index=False)

    ''' Extract testing and training MCC curves vs epoch '''
    epochs = EPOCHS_IN_LOG
    training = []; testing = []
    for line in open(log_file, 'r+'):
        if 'Training MCC' in line:
            training.append( float(line.split()[2]) )
        elif 'Testing MCC' in line:
            testing.append( float(line.split()[2]) )
    df = pd.DataFrame(columns=['epoch', 'A_test_mcc', 
                       'B_test_mcc', 'C_test_mcc', 
                       'D_test_mcc', 'E_test_mcc',
                       'A_train_mcc', 'B_train_mcc',
                       'C_train_mcc', 'D_train_mcc',
                       'E_train_mcc'])
    df.loc[0] = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range(epochs):
        training_mccs = map(lambda x: training[i+epochs*x], range(len(MODELS)))
        testing_mccs = map(lambda x: testing[i+epochs*x], range(len(MODELS)))
        row = [i+1] + list(testing_mccs) + list(training_mccs)
        df.loc[i+1] = row
    df.to_csv(out_fit_file, sep=',', index=False)
  
    
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
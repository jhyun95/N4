# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:16:59 2018

@author: jhyun_000
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
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
#    fig, ax = plt.subplots(1, 1, figsize=(6,4))
#    ax.scatter(pos_ints, neg_ints, label='images')
#    ax.scatter([REAL_PINT], [REAL_NINT], label='yeast')
#    ax.set_xlabel('% of pairs with positive interactions')
#    ax.set_ylabel('% of pairs with negative interactions')
#    plt.legend(loc='upper right')  
    
if __name__ == '__main__':
    main()
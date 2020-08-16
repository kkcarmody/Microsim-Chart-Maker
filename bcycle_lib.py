import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import re

# Business cycles time series
def drop_empty_rows(df):
    ''' Takes a dataframe and drops rows with all zeroes.
    '''
    return df.loc[(df != 0).any(axis = 1)]

def read_bcycle_data(cache_dirs):
    ''' Takes list of cache directory paths, reads in bcycle dataframe, prints number of BusinessCycle directories in the cache (should equal 0)
    '''
    bcycle_vars = ['PC' + str(i) for i in range(1, 9)]
    bcycle_path_list = [s for s in cache_dirs if 'BusinessCycle' in str(s)]
    print('Number of bcycle directories (should = 1): ' + str(len(bcycle_path_list)))
    bcycle_path = bcycle_path_list[0]
    
    bcycle_df = drop_empty_rows(pd.read_pickle(bcycle_path / 'cycle_factors.pkl')[bcycle_vars])
    bcycle_df.loc[2020] = np.nan

    return bcycle_df


def bcycle_subplots(bcycle_df, bcycle_output_path, show_plots = False):
    ''' Takes dataframe bcycle and generates 2x4 subplots of PCs
    '''
    palette = sns.color_palette()

    fig = plt.figure(figsize = (10, 5))

    num = 0
    for column in bcycle_df:
        num += 1
        plt.subplot(2, 4, num)

        for v in bcycle_df:
            plt.plot(bcycle_df.index, bcycle_df[v], marker = '', color = 'grey', linewidth = 0.6, alpha = 0.3)

        plt.plot(bcycle_df.index, bcycle_df[column], marker = '', color = palette[num], linewidth = 2.4, alpha = 1, label = column)

        plt.xlim(1960, 2020)
        plt.ylim(-20, 10)

        if num in range(5):
            plt.xticks([])
        if num not in [1, 5]:
            plt.yticks([])
        
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette[num])

    plt.text(1885, -29, 'Year', ha = 'center', va = 'center', fontsize = 12)
    plt.savefig(bcycle_output_path / 'bcycle.png')

    if show_plots:
        plt.show()

    plt.clf()
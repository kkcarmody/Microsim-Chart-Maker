import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import re
import pickle

from macro_lib import plot_ts

from dicts import slice_dict
from dicts import var_label_dict


def read_micro_data(target_run_path):
    ''' Given target run path, reads in micro data to dataframe
    '''
    return pd.read_pickle(target_run_path / 'micro.pkl.gz')


def make_micro_dfs(run_dirs, n_runs = 2, pct_sample = 1):
    ''' Constructs dictionary of n runs
    '''
    micro_dfs = {}
    for i, run in enumerate(run_dirs[:n_runs]):
        micro_dfs['run' + str(i)] = read_micro_data(run).sample(frac = pct_sample)
    
    return micro_dfs


def make_means(df, vars, groups = ['Year']):
    ''' Makes dataframe of a variable's mean by a set of groups
    '''
    return df.groupby(by = groups).mean()[vars].reset_index(inplace = False)


def make_dfs_means(df_dict, vars, groups = ['Year']):
    dfs_means = {}
    for key in df_dict:
        dfs_means[key] = make_means(df_dict[key], vars, groups = groups)
    return dfs_means


def cache_dict(df_dict, output_path, filename):
    output = open(output_path / filename, 'wb')
    pickle.dump(df_dict, output)
    output.close()


def cache_dict_means(df_dict, vars, output_path, groups = ['Year']):
    df_dicts = make_dfs_means(df_dict, vars, groups)
    filename = '_'.join(('micro', str(vars), '.'.join(groups))) + '.pkl'
    cache_dict(df_dicts, output_path, filename)
    return filename


def slice_df_dict(df_dict, vars, condition_var, condition_greater = 'None', condition_isin = 'None'):
    ''' Takes dictionary of dfs and slices according to either condition_var == condition_equal or condition_var > condition_great. Returns dictionary of sliced dfs for specified variables.
    '''
    sliced_df_dict = {}
    for key in df_dict:
        if condition_greater != 'None':
            sliced_df_dict[key] = df_dict[key].loc[df_dict[key][condition_var] > condition_greater, :][vars]

        if condition_isin != 'None':
            sliced_df_dict[key] = df_dict[key].loc[df_dict[key][condition_var].isin(condition_isin), :][vars]
    
    return sliced_df_dict


def cache_sliced_dfs_means(df_dict, var, groups, output_path, condition_var, condition_greater = 'None', condition_isin = 'None'):
    vars = [var] + groups
    if condition_greater != 'None':
        sliced_dfs = slice_df_dict(df_dict, vars, condition_var, condition_greater = condition_greater)
    
    elif condition_isin != 'None':
        sliced_dfs = slice_df_dict(df_dict, vars, condition_var, condition_isin = condition_isin)

    else:
        sliced_dfs = df_dict
    
    return cache_dict_means(sliced_dfs, var, output_path, groups)


def rename_categories(df, var, var_label_dict):
    dict = var_label_dict[var]
    newcols = []
    for col in df.columns:
        if var not in col:
            newcols.append(col)
        if var in col:
            key = int(col[-1])
            newvar = col.replace(col[-1], dict[key])
            newcols.append(newvar)
    
    df.columns = newcols


def slice_categories(df_dict, var, groups, var_label_dict):
    vars = [var] + groups

    df_dict_sliced = {}
    for key in df_dict:
        df_dict_sliced[key] = pd.get_dummies(df_dict[key][vars], columns = [var])

        rename_categories(df_dict_sliced[key], var, var_label_dict)

    return df_dict_sliced


# Read data and make charts:

def read_cache_dict(filepath):
    pkl_file = open(filepath, 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    return mydict


def swap_var_values(df):
    df['Education'].replace({2 : 1, 3 : 1, 4 : 2, 5 : 2}, inplace = True)
    df['ClassOfWorker'].replace({2 : 1, 3 : 2, 4 : 3, 5 : 3}, inplace = True)


def label_var_values(df, vars):
    for var in vars:
        df.replace({var : var_label_dict[var]}, inplace = True)
    return df


## Categorical:

def plot_lines(dfs, var, xvar, facet, units, ymax, show_plots = False, output_path = False, alpha = 0.35, linewidth = 1.2, values = 'None', projection = True, title = False, smoothing = False):
    facet_values = dfs['run0'][facet].drop_duplicates().sort_values()

    palette = sns.color_palette()
    fig = plt.figure(figsize = (10, 6))

    for key in dfs.keys():
        vars = [var for var in dfs[key].columns if var != xvar and var!= facet]

        for i, value in enumerate(facet_values):
            df = dfs[key]

            plt.subplot(1, len(facet_values), i + 1)

            df = df.loc[df[facet] == value, :]

            for num, var in enumerate(vars):
                if smoothing:
                    yvar = df[var].rolling(smoothing).mean()
                else:
                    yvar = df[var]

                plt.plot(df[xvar], yvar, marker = '', color = palette[num], linewidth = linewidth, alpha = alpha, label = var)

            plt.ylim(0, ymax)

            if xvar == 'Age':
                plt.xlim(16, 90)
                plt.autoscale(False)

            if 'Died' in values:
                plt.xlim(40, 70)
                plt.autoscale(False)

            if 'Birth' in values:
                plt.xlim(20, 50)
                plt.autoscale(False)

            if 'Children' in values:
                plt.xlim(20, 70)

            if i != 0:
                plt.yticks([])

            if key == 'run0' and projection:
                plt.axvline(x = 2019, color = 'lightgrey', linestyle = 'dashed', label = 'Start of projection')
        
            plt.title(facet + ' == ' + var_label_dict[facet][value], loc = 'left', fontsize = 12)

        if key == 'run0':
            leg = plt.legend(loc = 'best', prop = {'size':10})
            for lh in leg.legendHandles:
                lh.set_alpha(1)

    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)
    plt.xlabel(xvar)
    plt.ylabel(units)
    plt.ylim(0, ymax)

    if output_path and values == 'None':
        plt.savefig(output_path / '_'.join((var, 'share by', xvar, facet)))

    if output_path and values != 'None':
        plt.savefig(output_path / '_'.join((values, 'share by', values, xvar, facet)))

    if show_plots:
        plt.show()


## Simple Means:
def plot_simplemeans(dfs, var, chart_dict, ymax, show_plots = False, output_path = False, title = True):
    units = chart_dict['Simple Means'][var]['units']
    xvar = chart_dict['Simple Means'][var]['xvar']
    facet = chart_dict['Simple Means'][var]['facet']
    cat = chart_dict['Simple Means'][var]['cat']
    smoothing = chart_dict['Simple Means'][var]['smoothing']

    vars = [v for v in dfs['run0'].columns]

    if facet == 'Year' or cat == 'Year':
        dfs = slice_df_dict(dfs, vars, condition_var = 'Year', condition_isin = [2000, 2015, 2030, 2045])

    if xvar == 'Year':
        projection = True
    else:
        projection = False

    for key in dfs.keys():       
        dfs[key] = dfs[key].pivot_table(columns = cat, values = var, index = [xvar, facet], aggfunc = 'sum').reset_index()
        
        newcols = []
        for col in dfs[key].columns:
            if not isinstance(col, int):
                newcols.append(col)
            else:
                newvar = var_label_dict[cat][col]
                newcols.append(newvar)
        
        dfs[key].columns = newcols
        dfs[key] = dfs[key].dropna()

    plot_lines(dfs, cat, xvar, facet, units, ymax, projection = projection, show_plots = show_plots, output_path = output_path, title = title, values = var, smoothing = smoothing)
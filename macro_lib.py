import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import re

def read_macro_data(target_run_path):
    ''' Given target run path, reads in macro data to dataframe
    '''
    macro_df = pd.read_pickle(target_run_path / 'macro.pkl.gz')
    macro_df.loc[0, 'CapitalElasticity'] = macro_df.loc[1, 'CapitalElasticity']
    return macro_df


def different_cols(df1, df2):
    ''' Returns columns which differ between two dataframes
    '''
    if all(df1.columns == df2.columns):
        return [var for var in df1.columns if all(df1[var] != df2[var])]
    else:
        raise ValueError('Dataframes must have the same columns')

def get_stochastic_vars(run_dirs):
    return different_cols(read_macro_data(run_dirs[0]), read_macro_data(run_dirs[1]))


def growth(ts):
    ''' Takes a time series and converts to growth rates
    '''
    return ts / ts.shift(1) - 1


def plot_ts(years, ts, units = 'Percent Annual Growth', color = 'C0', linewidth = 1, alpha = 1, ts_label = True, projection = True, projection_start = 2019, legend_loc = 'best'):
    ''' Plots a time series line chart
    '''
    if ts_label == True:
        plt.plot(years, ts, label = ts.name, color = color, linewidth = linewidth, alpha = alpha)
    else:
        plt.plot(years, ts, label = '', color = color, linewidth = linewidth, alpha = alpha)

    plt.xlabel('Year')
    plt.ylabel(units)
    
    if projection == True:
        plt.axvline(x = projection_start, color = 'lightgrey', linestyle = 'dashed', label = 'Start of projection')
    
    leg = plt.legend(loc = legend_loc, prop = {'size':10})
    for lh in leg.legendHandles:
        lh.set_alpha(1)


def macro_plots(macro_vars, macro_df, macro_output_path, show_plots = False):
    ''' Takes list of variables and macro dataframe and generates time series charts
    '''
    for var in macro_vars:
        plot_ts(macro_df['Year'], growth(macro_df[var]), linewidth = 1.5)

        if macro_output_path:
            plt.savefig(macro_output_path / var)
        
        if show_plots:
            plt.show()

        plt.clf()


# Stochastic variables:
def aggregate_macro_data(run_dirs, var):
    ''' Takes list of run directories and aggregates vars to a single dataframe with first column 'var' and rest 'var1', 'var2', and so on.
    '''
    df = read_macro_data(run_dirs[0])[['Year', var]].rename(columns = {var : str(var + '_Simulations')})
    for i, path in enumerate(run_dirs):
        df2 = read_macro_data(path)[['Year', var]].rename(columns = {var : str(var + str(i + 1))})
        df = df.merge(df2, how = 'outer', on = 'Year')
    
    mean = df.melt(id_vars = 'Year').groupby('Year').mean()
    mean['value'] = growth(mean['value'])
    mean = mean.rename(columns = {'value' : str(var + '_Mean')})

    df = pd.merge(df, mean, how = 'outer', on = 'Year')
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    return df.set_index('Year')


def multiple_lineplot(df, macro_output_path = False, color = 'C0', linewidth = 0.6, alpha = 0.3, show_plots = False):
    for i, v in enumerate(df):
        if i == 0:
            plot_ts(df.index, df[v], linewidth = 1.5, alpha = .95, ts_label = True, projection = True)
        elif i == 1:
            plot_ts(df.index, df[v], linewidth = linewidth, alpha = alpha, ts_label = True, projection = False)
        else:
            plot_ts(df.index, df[v], linewidth = linewidth, alpha = alpha, ts_label = False, projection = False)

    if macro_output_path:
        plt.savefig(macro_output_path / df.columns[0])

    if show_plots:
        plt.show()

    plt.clf()


def make_macro_stochastic_plots(dirs, vars, output_path, show_plots = False):
    for var in vars:
        temp_df = aggregate_macro_data(dirs, var)
        temp_df1 = temp_df.iloc[:, 0]
        temp_df2 = temp_df.iloc[:, 1:].apply(growth)

        temp_df = pd.merge(temp_df1, temp_df2, how = 'outer', on = 'Year')

        multiple_lineplot(temp_df, output_path, show_plots = show_plots)
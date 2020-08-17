import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

from pathlib import Path
import re
import pickle

import macro_lib
import bcycle_lib
import micro_lib

import getpass
import time

from dicts import slice_dict
from dicts import var_label_dict
from dicts import chart_dict

time_start = time.time()

# Params:
## Output path:
user = getpass.getuser()
user_path = Path('C:/Users/' + user + '/')
output_path = Path(user_path / 'OneDrive - PennO365/Microsim Documentation/')
chart_cache_path = Path(output_path / 'Cache/')

## Display plots as they're made?
show_plots = True

## Microsim cache location
if user == 'KodyC':
    cache_path = Path('C:/Microsim Cache/')
elif user == 'kodyc':
    cache_path = Path('B:/Microsim Cache/')
else:
    raise Exception('Who are you')

cache_dirs = [x for x in cache_path.iterdir() if x.is_dir()]
run_dirs = [x for x in cache_dirs if 'MicrosimRunner' in str(x) and '5f78f6743c' not in str(x)]
target_run_path = run_dirs[0]

## Target paths for saving plots:
bcycle_output_path = output_path / 'Bcycle Charts'
macro_output_path = output_path / 'Macro Charts/'
micro_output_path = output_path / 'Micro Charts/'

## Microdata Parameters
cache_files = True
n_runs = 8
run_pct = 1
# n_runs = len(run_dirs)


# Business cycle plots
bcycle_df = bcycle_lib.read_bcycle_data(cache_dirs)

bcycle_lib.bcycle_subplots(bcycle_df, bcycle_output_path, show_plots = show_plots)

print('Done with BCycles!')


# Macro plots
macro_vars = ['CapitalElasticity', 'CapitalServices', 'GDPPriceIndexPrivate', 'GDPPrivate', 'Immigration', 'LaborInput', 'MultifactorProductivity', 'Population']

## Stochastic vars (use all runs):
macro_stochastic_vars = macro_lib.get_stochastic_vars(run_dirs)
# > Deaths, HourWorkedPrivate, Immigration, Population

macro_lib.make_macro_stochastic_plots(dirs = run_dirs, vars = macro_stochastic_vars, output_path = macro_output_path, show_plots = show_plots)


## Deterministic vars (only need one run):
macro_deterministic_vars = [item for item in macro_vars if item not in macro_stochastic_vars]
macro_df = macro_lib.read_macro_data(target_run_path)

macro_lib.macro_plots(macro_deterministic_vars, macro_df, macro_output_path, show_plots = show_plots)

print('Done with Macro!')


# Micro plots
micro_start = time.time()

years = [2000, 2015, 2030, 2045] # years to use for facets

if cache_files:
    micro_dfs = micro_lib.make_micro_dfs(run_dirs, n_runs, run_pct)
    micro_df = micro_dfs['run0']

    print('Imported ' + str(n_runs) + ', ' + str(100 * run_pct) + ' percent runs in ' + '--- %s seconds ---' % (time.time() - micro_start))

    for key in micro_dfs.keys():
        micro_lib.swap_var_values(micro_dfs[key])

    cache_file_dict_path = [x for x in chart_cache_path.iterdir() if 'filename' in str(x)][0]
    if cache_file_dict_path:
        cache_file_dict = micro_lib.read_cache_dict(cache_file_dict_path)
    else:
        cache_file_dict = {'Simple Means' : {}, 
                        'Categorical' : {}, 
                        'Workers' : {}}

    for key in slice_dict['Simple Means']:
        dict = slice_dict['Simple Means'][key]
        cache_file_dict['Simple Means'][key] = micro_lib.cache_sliced_dfs_means(micro_dfs, var = key, groups = dict['groups'], output_path = chart_cache_path, condition_var = dict['condition_var'], condition_greater = dict['condition_greater'], condition_isin = dict['condition_isin'])
        
        print('Done caching %s' % (key))

    for key in slice_dict['Categorical']:
        dict = slice_dict['Categorical'][key]
        sliced_dfs = micro_lib.slice_categories(micro_dfs, var = key, groups = dict['groups'], var_label_dict = var_label_dict)

        vars = [var for var in sliced_dfs['run0'].columns if key in var]

        cache_file_dict['Categorical'][key] = micro_lib.cache_dict_means(sliced_dfs, vars = vars, output_path = chart_cache_path, groups = dict['groups'])

        print('Done caching %s' % (key))
    
    micro_lib.cache_dict(cache_file_dict, chart_cache_path / 'filename_dict.pkl')

else:
    print('Skipping the micro cache process')
    cache_file_dict = micro_lib.read_cache_dict(chart_cache_path / 'filename_dict.pkl')

    micro_df = micro_lib.read_micro_data(target_run_path)
    micro_lib.swap_var_values(micro_df)


## Categorical Plots
for var in chart_dict['Categorical']:
    dfs = micro_lib.read_cache_dict(chart_cache_path / cache_file_dict['Categorical'][var])
    
    facet = chart_dict['Categorical'][var]['facet']
    
    units = 'Percent of Population'
    xvar = 'Year'
    ymax = chart_dict['Categorical'][var]['ymax']

    micro_lib.plot_lines(dfs, var, xvar, facet, units, ymax, show_plots = True, output_path = micro_output_path)
    

## Simple Means:
for var in chart_dict['Simple Means']:
    dfs = micro_lib.read_cache_dict(chart_cache_path / cache_file_dict['Simple Means'][var])

    ymax = chart_dict['Simple Means'][var]['ymax']

    print(var + ':')
    micro_lib.plot_simplemeans(dfs, var, chart_dict, ymax, show_plots = True, output_path = micro_output_path, title = var)
    

print('Done with micro! In' + '--- %s seconds ---' % (time.time() - micro_start))


## Work:
sample = micro_df.sample(50000)
sample = micro_lib.label_var_values(sample, ['Gender', 'Education', 'Married', 'ClassOfWorker'])


plot = sns.lmplot('Age', 'logCoreWageBase', data = sample[sample['logCoreWageBase'] > 0], col = 'Gender', hue = 'Education', lowess = True, scatter_kws = {'alpha' : 0.1, 's' : 4})
plot._legend.remove()
leg = plt.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)
plot.savefig(micro_output_path / 'CoreWageBase_Age_Gender_Education')


plot = sns.lmplot('Age', 'logCoreWageBase', data = sample[sample['logCoreWageBase'] > 0], col = 'Married', hue = 'Gender', lowess = True, scatter_kws = {'alpha' : 0.1, 's' : 2})
plot._legend.remove()
leg = plt.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)
plot.savefig(micro_output_path / 'CoreWageBase_Age_Gender_Married')


plot = sns.lmplot('Age', 'logCoreWageShock', data = sample[sample['logCoreWageShock'] != -1], col = 'Gender', hue = 'Education', lowess = True, scatter_kws = {'alpha' : 0.1, 's' : 2})
plot._legend.remove()
leg = plt.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.ylim(-2, 2)
plot.savefig(micro_output_path / 'CoreWageShock_Age_Gender_Education')


plot = sns.lmplot('Age', 'HoursWorked', data = sample.loc[sample['Worker'] == 1], col = 'Gender', hue = 'Education', lowess = True, scatter_kws = {'alpha' : 0.1, 's' : 2})
plot._legend.remove()
leg = plt.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)
plot.savefig(micro_output_path / 'HoursWorked_Age_Gender_Education')


plot = sns.lmplot('Age', 'HoursWorked', data = sample.loc[sample['Worker'] == 1], col = 'Gender', hue = 'Married', lowess = True, scatter_kws = {'alpha' : 0.1, 's' : 2})
plot._legend.remove()
leg = plt.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)
plot.savefig(micro_output_path / 'HoursWorked_Age_Gender_Married')

##-------------------------------------------------------------------




print('All done! :) used ' + str(n_runs) + ' runs in ' + '--- %s seconds ---' % (time.time() - time_start))
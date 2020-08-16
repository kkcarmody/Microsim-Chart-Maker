import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import pickle

from dicts import *
import getpass

import time

user = getpass.getuser()
user_path = Path('C:/Users/' + user + '/')
output_path = Path(user_path / 'OneDrive - PennO365/Microsim Documentation/')
micro_output_path = output_path / 'Micro Charts/'

## Microsim cache location
if user == 'KodyC':
    cache_path = Path('C:/Microsim Cache/')
elif user == 'kodyc':
    cache_path = Path('B:/Microsim Cache/')
else:
    raise Exception('Who are you')

cache_dirs = [x for x in cache_path.iterdir() if x.is_dir()]
run_dirs = [x for x in cache_dirs if 'MicrosimRunner' in str(x) and '5f78f6743c' not in str(x)]


categorical_vars = ['Race', 'LegalStatus', 'ClassOfWorker']
means_vars = ['NaturalizedThisYear', 'OverstayedVisaThisYear', 'EmigratedThisYear', 'Married', 'EducationYears', 'DiedThisYear', 'GaveBirthThisYear', 'WorkDisability', 'Worker', 'ChildrenUnder18']


def read_micro_data(target_run_path):
    return pd.read_pickle(target_run_path / 'micro.pkl.gz')

def swap_var_values(df):
    df['Education'].replace({2 : 1, 3 : 1, 4 : 2, 5 : 2}, inplace = True)
    df['ClassOfWorker'].replace({2 : 1, 3 : 2, 4 : 3, 5 : 3}, inplace = True)


class ChartMaker:
    def __init__(self, var):
        ''' Pulls class attributes from dictionaries
        '''
        self.var = var
        self.condition_var = None
        for current_dict in [slice_dict, chart_dict]:
            print(current_dict.keys(), var, current_dict[var])
            for key in current_dict.get(var).keys():
                setattr(self, key, current_dict.get(var).get(key))

    def get_data(self, run_dirs):
        ''' Reads in microsim runs, subsetted to relevant variables; returns dictionary of dataframes
        '''
        n_runs = 3
        pct_sample = 1

        vars_to_keep = set(self.groups + [self.var, self.condition_var])
        vars_to_keep.discard(None)
        start = time.time()

        dfs = {}
        for i, run in enumerate(run_dirs[:n_runs]):
            key = 'run' + str(i)
            dfs[key] = read_micro_data(run).sample(frac = pct_sample)
            swap_var_values(dfs[key]) # swap values of education and class of worker
            dfs[key] = dfs[key][vars_to_keep] # subset to vars we will use
        
        print('Imported ' + str(n_runs) + ', ' + str(100 * pct_sample) + ' percent runs in ' + '--- %s seconds ---' % (time.time() - start))
        
        return dfs

    def rename_categories(self, df):
        current_dict = var_label_dict[self.var]
        newcols = []
        for col in df.columns:
            if self.var not in col:
                newcols.append(col)
            if self.var in col:
                key = int(col[-1])
                newvar = col.replace(col[-1], current_dict[key])
                newcols.append(newvar)
        
        df.columns = newcols

    def reshape_categories(self, dfs):
        dfs_sliced = {}
        for key in dfs:
            dfs_sliced[key] = pd.get_dummies(dfs[key][[self.var] + self.groups], columns = [self.var])

            self.rename_categories(dfs_sliced[key])

        return dfs_sliced


def make_categorical_charts():
    var = 'Race'

    chart_maker = ChartMaker(var)
    dfs = chart_maker.get_data(run_dirs)
    dfs_reshaped = chart_maker.reshape_categories(dfs)
    
    plot_lines(dfs_reshaped, chart_maker.var, chart_maker.xvar, chart_maker.facet, chart_maker.units, chart_maker.ymax, show_plots = True, output_path = False)

make_categorical_charts()


def plot_lines(self):
    return 'TODO'


    # Cat: micro_lib.plot_lines(dfs, var, xvar, facet, units, ymax, show_plots = True, output_path = micro_output_path)

    # Simple means: plot_lines(dfs, cat, xvar, facet, units, ymax, projection = projection, show_plots = show_plots, output_path = output_path, title = title, values = var, smoothing = smoothing)
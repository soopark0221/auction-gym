import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from os import path

data_path = 'results/results_to_be_plotted'
output_dir = 'results/'
FIGSIZE = (8, 5)
FONTSIZE = 14
rounds_per_iter = 10000
obs_embedding_size = 5
embedding_size = 5

def plot_measure_overall(df, measure_name='Social Surplus'):
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time: {experiment}', fontsize=FONTSIZE + 2)
        sns.lineplot(data=df, x="Iteration", hue='Model', y=measure_name, ax=axes)
        min_measure = min(0.0, np.min(df[measure_name]))
        max_measure = max(0.0, np.max(df[measure_name]))
        plt.xlabel('Iteration', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        factor = 1.1 if min_measure < 0 else 0.9
        plt.ylim(min_measure * factor, max_measure * 1.1)
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{experiment}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs_{obs_embedding_size}_emb_of_{embedding_size}.png", bbox_inches='tight')
        return df

if __name__=='__main__':
    experiment = 'DR_PO'

    data_path = path.join(data_path, experiment)
    settings = ['Baseline', 'Gaussian_Noise', 'NoisyNet', 'SWAG']
    file_list = [path.join(data_path, setting+'.csv') for setting in settings]

    df_list = []

    for setting, file in zip(settings, file_list):
         df = pd.read_csv(file)
         df = df[df['Iteration']<15]
         df['Model'] = setting
         df_list.append(df)
    
    df = pd.concat(df_list)
    df = df[df['Measure Name']=='Social Surplus']
    df = df[['Run', 'Iteration', 'Measure', 'Model']]
    columns = ['Run', 'Iteration', 'Social Surplus', 'Model']
    df.columns = columns
    num_iter = np.max(df['Iteration'])
    num_runs = np.max(df['Run'])

    plot_measure_overall(df)
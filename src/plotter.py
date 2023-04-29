import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from os import path

data_path = '/home/soopark0221/auction-gym/results/results_to_plot'
output_dir = '/home/soopark0221/auction-gym/results/plots'
experiment = 'Eps'
Bid = 'SB'
subpath = '1'
measure_to_plot = 'Net Utility'
FIGSIZE = (8, 5)
FONTSIZE = 14

def plot_measure_overall(df, measure_name):
    df = df[~df['Agent'].str.startswith('Competitor')]
    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title(f'{measure_name} Over Time: {experiment}', fontsize=FONTSIZE + 2)
    sns.lineplot(data=df, x="Step", hue="Agent", y=measure_name, ax=axes)
    min_measure = min(0.0, np.min(df[measure_name]))
    max_measure = max(0.0, np.max(df[measure_name]))
    plt.xlabel('Step', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
    factor = 1.1 if min_measure < 0 else 0.9
    plt.ylim(min_measure * factor, max_measure * 1.1)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_{measure_name}.png", bbox_inches='tight')
    return df

def plot_vector_measure_per_agent(df, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        df = df[~df['Agent'].str.startswith('Competitor')]

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Distribution', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.boxplot(data=df, x="Iteration", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel('Iteration', fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
            # plt.ylim(min_measure * factor, max_measure * 1.1)
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.legend(loc='lower right', fontsize=FONTSIZE-4, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{experiment}_{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
        # plt.show()
        return df

def utility_per_method():
    global experiment, data_path
    subpath = 'utility_1'
    data_path = path.join(data_path, subpath)
    methods = ['Logistic-Eps', 'Logistic-UCB', 'NeuralNet', 'NeuralLogistic-Eps', 'NeuralLogistic-UCB']
    file_list = [path.join(data_path, method+'.csv') for method in methods]

    df_list = []

    for setting, file in zip(methods, file_list):
         df = pd.read_csv(file)
         df_list.append(df)
    
    df = pd.concat(df_list)
    df = df[~df['Agent'].str.startswith('Competitor')]

    df['Cumulative Utility'] = df.groupby(['Agent', 'Run'])['Net Utility'].cumsum()

    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title(f'Cumulative Utility Over Time: {experiment}', fontsize=FONTSIZE + 2)
    sns.lineplot(data=df, x="Step", hue="Agent", y="Cumulative Utility", ax=axes)
    min_measure = min(0.0, np.min(df["Cumulative Utility"]))
    max_measure = max(0.0, np.max(df["Cumulative Utility"]))
    plt.xlabel('Step', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.ylabel('Cumulative Utility', fontsize=FONTSIZE)
    factor = 1.1 if min_measure < 0 else 0.9
    plt.ylim(min_measure * factor, max_measure * 1.1)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_Cumulative_Utility.png", bbox_inches='tight')

def regret_per_method():
    global experiment, data_path
    subpath = 'regrets_2'
    data_path = path.join(data_path, subpath)
    methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.0_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.0_50']
    file_list = [path.join(data_path, method+'.csv') for method in methods]

    df_list = []

    for setting, file in zip(methods, file_list):
         df = pd.read_csv(file)
         df_list.append(df)
    
    df = pd.concat(df_list)
    df = df[~df['Agent'].str.startswith('Competitor')]

    for r in ['Allocation Regret', 'Overbid Regret', 'Underbid Regret']:
        df[r] = df.groupby(['Agent', 'Run'])[r].cumsum()
    for r in ['Allocation Regret', 'Overbid Regret', 'Underbid Regret']:
        plot_measure_overall(df, r)

def winning_prob_per_method():
    global experiment, data_path
    subpath = 'winning_prob_1'
    data_path = path.join(data_path, subpath)
    methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.0_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.0_50']
    file_list = [path.join(data_path, method+'.csv') for method in methods]

    df_list = []

    for setting, file in zip(methods, file_list):
         df = pd.read_csv(file)
         df_list.append(df)
    
    df = pd.concat(df_list)
    df = df[~df['Agent'].str.startswith('Competitor')]

    plot_measure_overall(df, 'Probability of winning')


def regret():
    global experiment,data_path
    subpath = 'regrets_1'
    data_path = path.join(data_path, subpath)
    methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        df = df.reset_index()
        df['methods'] = setting
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index()

    sns.lineplot(data=df, x="Step", y="Regret", ax=axes, hue='methods')

    plt.title(f'Regret Over Time: {experiment}', fontsize=FONTSIZE + 2)
    plt.ylabel('Regret', fontsize=FONTSIZE)

    plt.xlabel('x500 steps', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{Bid}_Regret.png", bbox_inches='tight')


def winning_prob_per_method():
    global experiment, data_path
    subpath = 'winning_prob_1'
    data_path_copy = path.join(data_path, subpath)
    methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    file_list = [path.join(data_path_copy, method+'.csv') for method in methods]

    df_list = []
    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
         df = pd.read_csv(file)
         df = df.reset_index()
         df_list.append(df)
    
    df = pd.concat(df_list)
    df = df[~df['Agent'].str.startswith('Competitor')]
    df = df.reset_index()

    plt.title(f'Winning Prob Over Time: {experiment}', fontsize=FONTSIZE + 2)
    sns.lineplot(data=df, x="Step", hue="Agent", y="Probability of Winning", ax=axes)

    plt.xlabel('Step', fontsize=FONTSIZE)
    plt.ylabel("Winning Prob", fontsize=FONTSIZE)

    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{Bid}_WinningProb.png", bbox_inches='tight')

def multibid():
    global experiment,data_path
    subpath = 'multibid_1'
    data_path_copy = path.join(data_path, subpath)
    methods = ['Eps0.0', 'Eps0.1', 'TS0.1', 'TS1.0','UCB0.1','UCB1.0']
    file_list = [path.join(data_path_copy, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)
    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        df = df.reset_index()
        df['methods'] = setting
        df = df[df['Auction']==3]
        df['Regret(cum)'] = df.groupby(['Run', 'Auction'])["Regret(500steps)"].cumsum()
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.reset_index()
    sns.lineplot(data=df, x="Step", y="Regret(cum)", ax=axes, hue='methods')

    plt.title(f'Regret Over Time: {experiment} ', fontsize=FONTSIZE + 2)
    plt.ylabel('Regret', fontsize=FONTSIZE)

    plt.xlabel('x500 steps', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_Regret_Multi.png", bbox_inches='tight')


if __name__=='__main__':
    #utility_per_method()
    #regret_per_method()
    #winning_prob_per_method()
    #regret()
    multibid()
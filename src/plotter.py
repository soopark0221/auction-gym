import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import os
from os import path
import json

data_path = 'results/results_to_plot'
output_dir = 'results/plots'
experiment = '300 agents'
agent = 'multi'
Bid = ''
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


def avg_regret():
    global experiment,data_path, agent
    subpath = 'regret'
    data_path1 = path.join(data_path, subpath)
    methods = [f'{Bid}Eps_0.0_m', f'{Bid}Eps_0.1_m', f'{Bid}TS_1.0_m', f'{Bid}UCB_10.0_m']
    #methods = ['SB_Eps_0.0', 'SB_Eps_0.1', 'SB_TS_1.0', 'SB_UCB_10.0']
    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        #df = df.reset_index()
        df['methods'] = setting
        df['avg_regret'] = df['Regret']/(df['Step']+1)
        if agent == 'single':
            df['Step'] = df['Step']*300
        
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index()

    sns.lineplot(data=df, x="Step", y="avg_regret", ax=axes, hue='methods')

    plt.title(f'Avg Regret Over Time: {experiment}', fontsize=FONTSIZE + 2)
    plt.ylabel('Avg Regret', fontsize=FONTSIZE)

    if agent == 'multi':
        plt.xlabel('steps per agent', fontsize=FONTSIZE)
    else:
        plt.xlabel('steps', fontsize=FONTSIZE)
    
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_AVG_Regret.png", bbox_inches='tight')

def cum_regret():
    global experiment,data_path, agent
    subpath = 'regret'
    data_path1 = path.join(data_path, subpath)
    methods = [f'{Bid}Eps_0.0_m', f'{Bid}Eps_0.1_m', f'{Bid}TS_1.0_m', f'{Bid}UCB_10.0_m']
    #methods = ['SB_Eps_0.0', 'SB_Eps_0.1', 'SB_TS_1.0', 'SB_UCB_10.0']
    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        df = df.reset_index()
        df['methods'] = setting
        if agent == 'single':
            df['Step'] = df['Step']*300
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index()

    sns.lineplot(data=df, x="Step", y="Regret", ax=axes, hue='methods')

    plt.title(f'Cumulative Regret Over Time: {experiment}', fontsize=FONTSIZE + 2)
    plt.ylabel('Cumulative Regret', fontsize=FONTSIZE)

    if agent == 'multi':
        plt.xlabel('steps per agent', fontsize=FONTSIZE)
    else:
        plt.xlabel('steps', fontsize=FONTSIZE)

    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_Cum_Regret.png", bbox_inches='tight')


def regret_ex():
    global experiment,data_path
    subpath = 'regret'
    data_path1 = path.join(data_path, subpath)
    methods = ['TS_0.1', 'TS_0.5', 'TS_1.0', 'TS_5.0', 'TS_10.0']
    #methods = ['SB_Eps_0.0', 'SB_Eps_0.1', 'SB_TS_1.0', 'SB_UCB_10.0']
    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        #df = df.reset_index()
        df['methods'] = setting

        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index()

    sns.lineplot(data=df, x="Step", y="Regret", ax=axes, hue='methods')

    plt.title(f'Cumulative Regret Over Time: {experiment}', fontsize=FONTSIZE + 2)
    plt.ylabel('Cumulative Regret', fontsize=FONTSIZE)

    plt.xlabel('steps per agent', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_Total_Regret.png", bbox_inches='tight')



def optimal():
    global experiment,data_path,agent
    subpath = 'optimalselection'
    data_path1 = path.join(data_path, subpath)
    methods = [f'{Bid}Eps_0.0_m', f'{Bid}Eps_0.1_m', f'{Bid}TS_1.0_m', f'{Bid}UCB_10.0_m']
    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        #df = df.reset_index()
        df['methods'] = setting
        if agent == 'single':
            df['Step'] = df['Step']*300
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index()

    sns.lineplot(data=df, x="Step", y="Optimal Selection Rate", ax=axes, hue='methods')

    plt.title(f'Selection Rate of Optimal Ads Over Time: {experiment}', fontsize=FONTSIZE + 2)
    plt.ylabel('Rate', fontsize=FONTSIZE)


    if agent == 'multi':
        plt.xlabel('steps per agent', fontsize=FONTSIZE)
    else:
        plt.xlabel('steps', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_Total_optimal.png", bbox_inches='tight')



def probw():
    global experiment,data_path,agent
    subpath = 'probw'
    data_path1 = path.join(data_path, subpath)
    methods = [f'{Bid}Eps_0.0_m', f'{Bid}Eps_0.1_m', f'{Bid}TS_1.0_m', f'{Bid}UCB_10.0_m']
    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        #df = df.reset_index()
        df['methods'] = setting
        df = df[~df['Agent'].str.startswith('Competitor')]
        df = df[~df['Agent'].str.startswith('Optimal')]
        if agent == 'single':
            df['Step'] = df['Step']*300
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index()

    sns.lineplot(data=df, x="Step", y="Probability of Winning", hue='methods', ax=axes)

    plt.title(f'Probability of Winning Over Time: {experiment}', fontsize=FONTSIZE + 2)
    plt.ylabel('Prob', fontsize=FONTSIZE)


    if agent == 'multi':
        plt.xlabel('steps per agent', fontsize=FONTSIZE)
    else:
        plt.xlabel('steps', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_Total_Probw.png", bbox_inches='tight')



def util():
    global experiment,data_path
    subpath = 'utility'
    data_path1 = path.join(data_path, subpath)
    methods = [f'{Bid}Eps_0.0_m', f'{Bid}Eps_0.1_m', f'{Bid}TS_1.0_m', f'{Bid}UCB_10.0_m']
    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        #df = df.reset_index()
        df = df[~df['Agent'].str.startswith('Competitor')]
        df = df[~df['Agent'].str.startswith('Optimal')]

        df['methods'] = setting
        if agent == 'single':
            df['Step'] = df['Step']*300
        df_list.append(df)

    df = pd.concat(df_list)
    df = df.reset_index()

    sns.lineplot(data=df, x="Step", y="Utility (Cumulative)", ax=axes, hue='methods')

    plt.title(f'Cumulative Utility Over Time: {experiment}', fontsize=FONTSIZE + 2)
    plt.ylabel('Cumulative Utility', fontsize=FONTSIZE)


    if agent == 'multi':
        plt.xlabel('steps per agent', fontsize=FONTSIZE)
    else:
        plt.xlabel('steps', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment}_Total_Utility.png", bbox_inches='tight')




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
    subpath = 'regret'
    data_path_copy = path.join(data_path, subpath)
    methods = ['Eps_0.0', 'Eps_0.1', 'TS_1.0', 'UCB_10.0']
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
    plt.savefig(f"{output_dir}/{experiment}_Regrets.png", bbox_inches='tight')


def replace_df(data_name):
    global experiment,data_path
    subpath = data_name
    data_path1 = path.join(data_path, subpath)
    methods = [f'{Bid}Eps_0.0_2', f'{Bid}Eps_0.1_2', f'{Bid}TS_1.0_2', f'{Bid}UCB_10.0_2']
    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file in zip(methods, file_list):
        df = pd.read_csv(file)
        df['Run'] = df['Run'].replace([2], 5)
        df['Run'] = df['Run'].replace([1], 4)
        df['Run'] = df['Run'].replace([0], 3)
        df.to_csv(f'{data_path1}/{setting}.csv')
        #df = df.reset_index()

def concat_df(data_name):
    global experiment,data_path
    subpath = data_name
    data_path1 = path.join(data_path, subpath)
    methods = [f'{Bid}Eps_0.0', f'{Bid}Eps_0.1', f'{Bid}TS_1.0', f'{Bid}UCB_10.0']
    methods2 = [f'{Bid}Eps_0.0_2', f'{Bid}Eps_0.1_2', f'{Bid}TS_1.0_2', f'{Bid}UCB_10.0_2']
    methods3 = [f'{Bid}Eps_0.0_m', f'{Bid}Eps_0.1_m', f'{Bid}TS_1.0_m', f'{Bid}UCB_10.0_m']
    #methods = ['Eps_0.0', 'Eps_0.1', 'TS_1.0', 'UCB_10.0']
    #methods2 = ['Eps_0.0_2', 'Eps_0.1_2', 'TS_1.0_2', 'UCB_10.0_2']
    #methods3 = ['Eps_0.0_m', 'Eps_0.1_m', 'TS_1.0_m', 'UCB_10.0_m']

    #methods = [f'Logistic_Eps0.0_{Bid}', f'Logistic_Eps0.1_{Bid}', f'Logistic_TS0.1_{Bid}', f'Logistic_TS1.0_{Bid}', f'Logistic_UCB0.1_{Bid}',f'Logistic_UCB1.0_{Bid}']
    #methods = ['Logistic_Eps0.0_20', 'Logistic_Eps0.1_20', 'Logistic_Eps0.0_50', 'Logistic_Eps0.1_50']
    file_list = [path.join(data_path1, method+'.csv') for method in methods]
    file_list2 = [path.join(data_path1, method+'.csv') for method in methods2]

    df_list = []

    fig, axes = plt.subplots(figsize=FIGSIZE)

    for setting, file1, file2 in zip(methods3, file_list, file_list2):
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        concate_data = pd.concat([df1,df2])
        concate_data.to_csv(f'{data_path1}/{setting}.csv')


if __name__=='__main__':
    dir_list = os.listdir(data_path)

    regret_df = pd.DataFrame(columns=['bonus_factor', 'optimism_scale', 'overbidding_facotr', 'eq_winning_rate', 'overbidding_steps'])
    utility_df = pd.DataFrame(columns=['bonus_factor', 'optimism_scale', 'overbidding_facotr', 'eq_winning_rate', 'overbidding_steps'])
    prob_win_df = pd.DataFrame(columns=['bonus_factor', 'optimism_scale', 'overbidding_facotr', 'eq_winning_rate', 'overbidding_steps'])
    optimal_selection_df = pd.DataFrame(columns=['bonus_factor', 'optimism_scale', 'overbidding_facotr', 'eq_winning_rate', 'overbidding_steps'])

    experiment_list = []
    allocator_list = []
    exploration_list = []

    regret_list = []
    utility_list = []
    prob_winning_list = []
    optimal_selection_list = []

    for dir in dir_list:
        if dir.endswith('DS_Store'):
            continue
        directory = path.join(data_path, dir)
        experiment_list.append(directory)

        with open(path.join(directory, 'agent_config.json')) as f:
            agent_config = json.load(f)
        for a in agent_config['agents']:
            if not 'Competitor' in a['name']:
                agent_config = a
                break

        name = agent_config['allocator']['type']
        mode = agent_config['allocator']['kwargs']['mode']
        if "NeuralBootstrap" in name:
            if "TS" in mode:
                allocator_list.append(f"NeuBoot-TS({agent_config['allocator']['kwargs']['nu']})")
            elif "UCB" in mode:
                allocator_list.append(f"NeuBoot-UCB({agent_config['allocator']['kwargs']['nu']})")
        elif "NeuralAllocator" in name:
            if "Epsilon-greedy" in mode:
                allocator_list.append(f"Neural-Eps({agent_config['allocator']['kwargs']['nu']})")
            elif "TS" in mode:
                allocator_list.append(f"Neural-TS({agent_config['allocator']['kwargs']['nu']})")
            elif "UCB" in mode:
                allocator_list.append(f"Neural-UCB({agent_config['allocator']['kwargs']['nu']})")
                
        exploration = [agent_config['bonus_factor']]
        bidder_config = agent_config['bidder']
        exploration.extend([bidder_config['kwargs']['optimism_scale'],
                            bidder_config['kwargs']['overbidding_factor'],
                            bidder_config['kwargs']['eq_winning_rate'],
                            bidder_config['kwargs']['overbidding_steps'],])
        exploration_list.append(exploration)

        with open(path.join(directory, 'training_config.json')) as f:
            training_config = json.load(f)
        
        regret_list.append()
        utility_list.append(pd.read_csv(path.join(directory, 'net_utility.csv')))
        prob_winning_list.append(pd.read_csv(path.join(directory, 'winning_probability.csv')))
        optimal_selection_list.append(pd.read_csv(path.join(directory, 'optimal_selection_rate.csv')))

        regret = pd.read_csv(path.join(directory, 'regret.csv'))
        regret['bonus_factor', 'optimism_scale', 'overbidding_facotr', 'eq_winning_rate', 'overbidding_steps'] = exploration
        print(regret)

        

   # avg_regret()
   # cum_regret()
   # optimal()
   # probw()
   # util()
    # regret_ex()

    #replace_df('regret')
    #concat_df('regret')
    #replace_df('utility')
    #concat_df('utility')
    #replace_df('probw')
    #concat_df('probw')
    #replace_df('optimalselection')
    #concat_df('optimalselection')
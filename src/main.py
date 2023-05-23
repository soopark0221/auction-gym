import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import time

from Agent import Agent
from AuctionAllocation import * # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  
from BidderAllocation import * 


def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''


def parse_agent_config(path):
    with open(path) as f:
        config = json.load(f)

    output_dir = config['output_dir']

    # Expand agent-config if there are multiple copies
    agent_configs = []
    num_agents = 0
    for agent_config in config['agents']:
        if 'num_copies' in agent_config.keys():
            for i in range(1, agent_config['num_copies'] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy['name'] += f' {i}'
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    return agent_configs, output_dir

def draw_features(rng, num_runs, context_dim, feature_dim, agent_configs):
    run2agents2items = {}
    run2agents2item_values = {}
    run2bilinear_map = {}
    for run in range(num_runs):
        agents2items = {}
        for agent_config in agent_configs:
            temp = []
            for k in range(agent_config['num_items']):
                feature = rng.normal(0.0, 1.0, size=feature_dim)
                temp.append(feature)
            agents2items[agent_config['name']] = np.stack(temp)
        run2agents2items[run] = agents2items

        agents2item_values = {
            agent_config['name']: np.ones((agent_config['num_items'],))
            for agent_config in agent_configs
        }
        run2agents2item_values[run] = agents2item_values

        run2bilinear_map[run] = rng.normal(0.0, 1.0, size=(context_dim, feature_dim))

    return run2agents2items, run2agents2item_values, run2bilinear_map

def instantiate_agents(rng, agent_configs, agents2item_values, agents2item_features, bilinear_map, context_dim, update_interval):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              item_features=agents2item_features[agent_config['name']],
              item_values=agents2item_values[agent_config['name']],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng, item_features=agents2item_features[agent_config['name']]{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              context_dim = context_dim,
              update_interval=update_interval,
              bonus_factor=(agent_config['bonus_factor'] if 'bonus_factor' in agent_config.keys() else 0.0))
        for agent_config in agent_configs
    ]

    for agent in agents:
        agent.explore_then_commit = explore_then_commit
        if (not 'Competitor' in agent.name) and (not isinstance(agent.allocator, OracleAllocator)) and agent.allocator.mode=='UCB':
            assert agent.allocator.c == agent.bidder.optimism_scale
        if isinstance(agent.allocator, OracleAllocator):
            agent.allocator.set_CTR_model(bilinear_map)
        if isinstance(agent.allocator, NeuralAllocator):
            agent.allocator.initialize(agents2item_values[agent.name])
        try:
            agent.bidder.initialize(agents2item_values[agent.name])
        except:
            pass

    return agents


def instantiate_auction(rng, training_config, bilinear_map, agents2items, agents2item_values, agents, max_slots, context_dim, obs_context_dim, context_dist):
    return Auction(rng,
                    eval(f"{training_config['allocation']}()"),
                    agents,
                    bilinear_map,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    context_dim,
                    obs_context_dim,
                    context_dist,
                    training_config['num_participants_per_round'])


def simulation_run(run):

    for i in tqdm(np.arange(1, num_iter+1), desc=f'run {run}'):
        auction.simulate_opportunity()

        if i%record_interval==0:
            for agent_id, agent in enumerate(auction.agents):
                agent2net_utility[agent.name].append(agent.get_net_utility())

                agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
                agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
                agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

                agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
                agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

                agent2winning_prob[agent.name].append(agent.get_winning_prob())

                if not isinstance(agent.bidder, TruthfulBidder):
                    regret.append(auction.get_regret())
                    optimal_selection_rate.append(agent.get_optimal_selection_rate())
                    bidding_error.append(agent.get_bidding_error())
                    bidding_optimal.append(auction.get_optimal_bidding())
                    winrate_optimal.append(auction.get_winrate_optimal())
                    utility_optimal.append(auction.get_optimal_utility())
                    optimistic_CTR_ratio.append(agent.get_optimistic_CTR_ratio())

                best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
                agent2best_expected_value[agent.name].append(best_expected_value)
                agent.move_index()
            auction_revenue.append(auction.revenue)
            auction.clear_revenue()

def measure_per_agent2df(run2agent2measure, measure_name):
        df_rows = {'Run': [], 'Agent': [], 'Step': [], measure_name: []}
        for run, agent2measure in run2agent2measure.items():
            for agent, measures in agent2measure.items():
                for step, measure in enumerate(measures):
                    df_rows['Run'].append(run)
                    df_rows['Agent'].append(agent)
                    df_rows['Step'].append(step)
                    df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)
    
def vector_of_measure2df(run2vector_measure, measure_name):
    df_rows = {'Run': [], 'Step': [], 'Index':[], measure_name: []}
    for run, vector_measures in run2vector_measure.items():
        for step, vector in enumerate(vector_measures):
            for (index), measure in np.ndenumerate(vector):
                df_rows['Run'].append(run)
                df_rows['Step'].append(step)
                df_rows['Index'].append(index)
                df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)

def plot_measure_per_agent(run2agent2measure, measure_name, log_y=False, yrange=None, optimal=None):
    if type(run2agent2measure) != pd.DataFrame:
        df = measure_per_agent2df(run2agent2measure, measure_name)
    else:
        df = run2agent2measure
    df = df.reset_index(drop=True)
    try:
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.lineplot(data=df, x="Step", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        plt.xlabel(f"x{record_interval} steps", fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE-4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
    except Exception as e:
        print(run2agent2measure)
        print(run2agent2measure.index.duplicated())
        print('cannot plot', e)
    return df

def plot_vector_measure(run2vector_measure, measure_name, log_y=False, yrange=None, optimal=None):
    if type(run2vector_measure) != pd.DataFrame:
        df = vector_of_measure2df(run2vector_measure, measure_name)
    else:
        df = run2vector_measure

    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title(f'{measure_name} Distribution', fontsize=FONTSIZE + 2)
    min_measure, max_measure = 0.0, 0.0
    sns.boxplot(data=df, x="Step", y=measure_name, ax=axes)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.ylabel(measure_name, fontsize=FONTSIZE)
    plt.xlabel(f"x{record_interval} steps", fontsize=FONTSIZE)
    if optimal is not None:
        plt.axhline(optimal, ls='--', color='gray', label='Optimal')
        min_measure = min(min_measure, optimal)
    if log_y:
        plt.yscale('log')
    if yrange is None:
        factor = 1.1 if min_measure < 0 else 0.9
    else:
        plt.ylim(yrange[0], yrange[1])
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
    return df

def measure2df(run2measure, measure_name):
    df_rows = {'Run': [], 'Step': [], measure_name: []}
    for run, measures in run2measure.items():
        for iteration, measure in enumerate(measures):
            df_rows['Run'].append(run)
            df_rows['Step'].append(iteration)
            df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)

def plot_measure(run2measure, measure_name, hue=None):
    if type(run2measure) != pd.DataFrame:
        df = measure2df(run2measure, measure_name)
    else:
        df = run2measure
    df = df.reset_index(drop=True)

    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
    if hue is None:
        sns.lineplot(data=df, x="Step", y=measure_name, ax=axes)
    else:
        sns.lineplot(data=df, x="Step", y=measure_name, hue=hue, ax=axes)
    min_measure = min(0.0, np.min(df[measure_name]))
    max_measure = max(0.0, np.max(df[measure_name]))
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
    plt.xlabel(f"x{record_interval} steps", fontsize=FONTSIZE)
    factor = 1.1 if min_measure < 0 else 0.9
    plt.ylim(min_measure * factor, max_measure * 1.1)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
    return df
                

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    with open('config/training.json') as f:
        training_config = json.load(f)


    # Set up Random Number Generator
    rng = np.random.default_rng(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])
    torch.manual_seed(training_config['random_seed'])

    num_runs = training_config['num_runs']
    num_iter  = training_config['num_iter']
    record_interval = training_config['record_interval']
    update_interval = training_config['update_interval']

    # Max. number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    context_dim = training_config['context_dim']
    feature_dim = training_config['feature_dim']
    context_dist = training_config['context_distribution']
    explore_then_commit = training_config['explore_then_commit']

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse configuration file
    agent_configs, output_dir = parse_agent_config(args.config)
    for a in agent_configs:
        if not 'Competitor' in a['name']:
            bonus_factor = (a['bonus_factor'] if 'bonus_factor' in a.keys() else 0.0)
            if a['bidder']['type']!='OracleBidder' and a['bidder']['type']!='RichBidder':
                optimism_scale, overbidding_factor= \
                    a['bidder']['kwargs']['optimism_scale'], a['bidder']['kwargs']['overbidding_factor']
            else:
                optimism_scale, overbidding_factor = 0.0, 0.0
            agent_name = a['name'].split()[0]

    output_dir = output_dir + time.strftime('%y%m%d-%H%M%S') + f"{agent_name}_{bonus_factor}_{optimism_scale}_{overbidding_factor}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.copy(args.config, os.path.join(output_dir, 'agent_config.json'))
    shutil.copy('config/training.json', os.path.join(output_dir, 'training_config.json'))
    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2allocation_regret = {}
    run2agent2overbid_regret = {}
    run2agent2underbid_regret = {}
    run2agent2best_expected_value = {}

    run2agent2CTR_RMSE = {}
    run2agent2CTR_bias = {}
    run2agent2winning_prob = {}
    run2auction_revenue = {}
    run2winrate_estimation = {}

    run2regret = {}
    run2optimal_selection_rate = {}
    run2bidding_error = {}
    run2winrate_optimal = {}
    run2bidding_optimal = {}
    run2utility_optimal = {}
    run2optimistic_CTR_ratio = {}

    run2agents2items, run2agents2item_values, run2bilinear_map = draw_features(rng, num_runs, context_dim, feature_dim, agent_configs)

    # Repeated runs
    for run in range(num_runs):
        agents2items = run2agents2items[run]
        agents2item_values = run2agents2item_values[run]
        bilinear_map = run2bilinear_map[run]
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items, bilinear_map, context_dim, update_interval)
        auction  = instantiate_auction(
            rng, training_config, bilinear_map, agents2items, agents2item_values, agents, max_slots, context_dim, context_dim, context_dist)
        
        # Placeholders for summary statistics per run
        agent2net_utility = defaultdict(list)
        agent2allocation_regret = defaultdict(list)
        agent2overbid_regret = defaultdict(list)
        agent2underbid_regret = defaultdict(list)
        agent2best_expected_value = defaultdict(list)

        agent2CTR_RMSE = defaultdict(list)
        agent2CTR_bias = defaultdict(list)

        agent2winning_prob = defaultdict(list)

        auction_revenue = []
        winrate_estimation = []
        regret = []
        optimal_selection_rate = []
        bidding_error = []
        winrate_optimal = []
        bidding_optimal = []
        utility_optimal = []
        optimistic_CTR_ratio = []

        simulation_run(run)

        # Store
        run2agent2net_utility[run] = agent2net_utility
        run2agent2allocation_regret[run] = agent2allocation_regret
        run2agent2overbid_regret[run] = agent2overbid_regret
        run2agent2underbid_regret[run] = agent2underbid_regret
        run2agent2best_expected_value[run] = agent2best_expected_value

        run2agent2CTR_RMSE[run] = agent2CTR_RMSE
        run2agent2CTR_bias[run] = agent2CTR_bias

        run2agent2winning_prob[run] = agent2winning_prob
        run2auction_revenue[run] = auction_revenue
        run2regret[run] = regret
        run2optimal_selection_rate[run] = optimal_selection_rate
        run2bidding_error[run] = bidding_error
        run2winrate_optimal[run] = winrate_optimal
        run2bidding_optimal[run] = bidding_optimal
        run2utility_optimal[run] = utility_optimal
        run2optimistic_CTR_ratio[run] = optimistic_CTR_ratio


    plot_vector_measure(run2bidding_error, 'Bidding Error')
    plot_vector_measure(run2bidding_optimal, 'Bidding of Optimal Agent')
    optimal_selection_df = plot_measure(run2optimal_selection_rate, 'Optimal Selection Rate')

    optimal_selection_df['bonus_factor'] = bonus_factor
    optimal_selection_df['optimism_scale'], optimal_selection_df['overbidding_factor'] = \
        [optimism_scale, overbidding_factor]
    optimal_selection_df.to_csv(f'{output_dir}/{agent_name}_optimal_selection_rate.csv', index=False)
    
    utility_df = measure_per_agent2df(run2agent2net_utility, 'Utility')
    optimal_utility_df = measure2df(run2utility_optimal, 'Utility')
    optimal_utility_df['Agent'] = 'Optimal'
    utility_df = pd.concat([utility_df, optimal_utility_df]).sort_values(['Agent', 'Run', 'Step'])
    utility_df['Utility (Cumulative)'] = utility_df.groupby(['Agent', 'Run'])['Utility'].cumsum()
    plot_measure_per_agent(utility_df, 'Utility')
    plot_measure_per_agent(utility_df, 'Utility (Cumulative)')
    utility_df['bonus_factor'] = bonus_factor
    utility_df['optimism_scale'], utility_df['overbidding_factor']= \
        [optimism_scale, overbidding_factor]
    utility_df.to_csv(f'{output_dir}/{agent_name}_net_utility.csv', index=False)

    plot_measure_per_agent(run2agent2best_expected_value, 'Best Expected Value')

    allocation_regret_df = plot_measure_per_agent(run2agent2allocation_regret, 'Allocation Regret')
    allocation_regret_df.to_csv(f'{output_dir}/{agent_name}_allocation_regret.csv', index=False)
    overbid_regret_df = plot_measure_per_agent(run2agent2overbid_regret, 'Overbid Regret')
    overbid_regret_df.to_csv(f'{output_dir}/{agent_name}_overbid_regret.csv', index=False)
    underbid_regret_df = plot_measure_per_agent(run2agent2underbid_regret, 'Underbid Regret')
    underbid_regret_df.to_csv(f'{output_dir}/{agent_name}_underbid_regret.csv', index=False)

    plot_measure(run2auction_revenue, 'Auction Revenue')

    winning_prob_df = measure_per_agent2df(run2agent2winning_prob, 'Probability of Winning')
    optimal_winning_prob_df = measure2df(run2winrate_optimal, 'Probability of Winning')
    optimal_winning_prob_df['Agent'] = 'Optimal'
    winning_prob_df = pd.concat([winning_prob_df, optimal_winning_prob_df])
    plot_measure_per_agent(winning_prob_df, 'Probability of Winning')
    winning_prob_df['bonus_factor'] = bonus_factor
    winning_prob_df['optimism_scale'], winning_prob_df['overbidding_factor'] = \
        [optimism_scale, overbidding_factor]
    winning_prob_df.to_csv(f'{output_dir}/{agent_name}_winning_probability.csv', index=False)

    CTR_RMSE_df = plot_measure_per_agent(run2agent2CTR_RMSE, 'CTR RMSE')
    CTR_RMSE_df['bonus_factor'] = bonus_factor
    CTR_RMSE_df['optimism_scale'], CTR_RMSE_df['overbidding_factor'] = \
        [optimism_scale, overbidding_factor]
    CTR_RMSE_df.to_csv(f'{output_dir}/{agent_name}_CTR_RMSE.csv', index=False)

    CTR_df = plot_measure_per_agent(run2agent2CTR_bias, 'CTR Bias', optimal=1.0)
    CTR_df.rename(columns={'CTR Bias':'CTR'}, inplace=True)
    CTR_df = CTR_df[~CTR_df['Agent'].str.startswith('Competitor')]
    CTR_df.drop(columns=['Agent'])
    CTR_df['Expected or Optimistic'] = 'Expected CTR'
    optimistic_CTR_df = measure2df(run2optimistic_CTR_ratio, 'CTR')
    optimistic_CTR_df['Expected or Optimistic'] = 'Optimistic CTR'
    CTR_df = pd.concat([CTR_df, optimistic_CTR_df])
    CTR_df.to_csv(f'{output_dir}/{agent_name}_CTR.csv', index=False)
    plot_measure(CTR_df, 'CTR', hue='Expected or Optimistic')

    regret_df = measure2df(run2regret, f'Regret({record_interval}steps)')
    regret_df['Regret'] = regret_df.groupby(['Run'])[f'Regret({record_interval}steps)'].cumsum()
    plot_measure(regret_df, 'Regret')
    regret_df['bonus_factor'] = bonus_factor
    regret_df['optimism_scale'], regret_df['overbidding_factor'] = \
        [optimism_scale, overbidding_factor]
    regret_df.to_csv(f'{output_dir}/{agent_name}_regret.csv', index=False)
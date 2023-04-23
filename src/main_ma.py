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
import pickle

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

def draw_features(num_auctions, rng, num_runs, context_dim, agent_configs):
    auction2run2agents2items = {}
    auction2run2agents2item_values = {}
    for auct in range(num_auctions):
        run2agents2items = {}
        run2agents2item_values = {}
        for run in range(num_runs):
            agents2items = {}
            for agent_config in agent_configs:
                temp = []
                for k in range(agent_config['num_items']):
                    feature = rng.normal(0.0, 1.0, size=context_dim)
                    temp.append(feature / np.sqrt(np.sum(feature**2)))
                agents2items[agent_config['name']] = np.stack(temp)
            run2agents2items[run] = agents2items
            
            agents2item_values = {
                # agent_config['name']: rng.lognormal(0.1, 0.2, agent_config['num_items'])
                agent_config['name']: np.ones((agent_config['num_items'],))
                for agent_config in agent_configs
            }

            run2agents2item_values[run] = agents2item_values
        auction2run2agents2items[auct] = run2agents2items
        auction2run2agents2item_values[auct] = run2agents2item_values
    return auction2run2agents2items, auction2run2agents2item_values

def instantiate_agents(rng, agent_configs, obs_context_dim, update_interval, context_dist, random_bidding):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              num_items=agent_config['num_items'],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              context_dim = obs_context_dim,
              update_interval=update_interval,
              random_bidding = random_bidding,
              memory=('inf' if 'memory' not in agent_config.keys() else agent_config['memory']))
        for agent_config in agent_configs
    ]

    return agents


def instantiate_auction(rng, training_config, agents2items, agents2item_values, agents, max_slots, context_dim, obs_context_dim, context_dist):
    return Auction(rng,
                    eval(f"{training_config['allocation']}()"),
                    agents,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    context_dim,
                    obs_context_dim,
                    context_dist,
                    training_config['num_participants_per_round'])


def simulation_run(run, multi_auctions):

    for i in tqdm(np.arange(1, num_iter+1), desc=f'run {run}'):
        for j in range(len(multi_auctions)):
            multi_auctions[j].simulate_opportunity(auction_no=j)

            if i%record_interval==0:
                for agent_id, agent in enumerate(multi_auctions[j].agents):
                    auction2agent2net_utility[j][agent.name].append(agent.get_net_utility(j))

                    auction2agent2allocation_regret[j][agent.name].append(agent.get_allocation_regret(j))
                    auction2agent2overbid_regret[j][agent.name].append(agent.get_overbid_regret(j))
                    auction2agent2underbid_regret[j][agent.name].append(agent.get_underbid_regret(j))

                    auction2agent2CTR_RMSE[j][agent.name].append(agent.get_CTR_RMSE(j))
                    auction2agent2CTR_bias[j][agent.name].append(agent.get_CTR_bias(j))

                    auction2agent2winning_prob[j][agent.name].append(agent.get_winning_prob(j))
                    auction2agent2CTR[j][agent.name].append(agent.get_CTRs(j))

                    if not isinstance(agent.bidder, TruthfulBidder):
                        bidding[j].append(np.array(agent.get_bid()))
                        uncertainty[j].append(agent.get_uncertainty())
                        regret[j].append(auction.get_regret())
                        optimal_selection_rate[j].append(agent.get_optimal_selection_rate(j))
                        bidding_error[j].append(agent.get_bidding_error(j))
                        bidding_optimal[j].append(auction.get_optimal_bidding())
                        winrate_optimal[j].append(auction.get_winrate_optimal())
                        utility_optimal[j].append(auction.get_optimal_utility())
                        optimistic_CTR_ratio[j].append(agent.get_optimistic_CTR_ratio(j))

                    best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs[j]])
                    auction2agent2best_expected_value[j][agent.name].append(best_expected_value)
                    agent.move_index(j)
                auction_revenue.append(auction.revenue)
                auction.clear_revenue()
        # if i%int(num_iter/5)==0:
        #     plot_winrate(target_agent)

def plot_winrate(agent):
    global auction, obs_context_dim
    for i in range(2):
        context = auction.generate_context()
        obs_context = context[:obs_context_dim]
        if isinstance(agent.allocator, OracleAllocator):
            item, estimated_CTR = agent.select_item(context)
        else:
            item, estimated_CTR = agent.select_item(obs_context)
        value = agent.item_values[item]
        gamma = np.linspace(0.1, 1.5, 128).reshape(-1,1)
        x = np.concatenate([
            np.tile(obs_context, (128, 1)),
            np.tile(estimated_CTR*value, (128,1))*gamma
        ], axis=1)
        x = torch.Tensor(x).to(agent.bidder.device)
        y = agent.bidder.winrate_model(x).numpy(force=True).reshape(-1,1)
        winrate_estimation.append(np.concatenate([gamma, y], axis=1))

def measure_per_agent2df(run2agent2measure, measure_name):
        df_rows = {'Run': [], 'Auction': [], 'Agent': [], 'Step': [], measure_name: []}
        for run, agent2measure in run2agent2measure.items():
            for auction, agent in agent2measure.items():
                for agent, measures in agent2measure.items():
                    for step, measure in enumerate(measures):
                        df_rows['Run'].append(run)
                        df_rows['Auction'].append(auction)
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

def full_measure2df(run2measure, measure_name):
    df_rows = {'Run': [], 'Step': [], 'Parameter':[], measure_name: []}
    for run, measure in run2measure.items():
        for step, measure_per_step in enumerate(measure):
            for index, measure_point in enumerate(measure_per_step):
                df_rows['Run'].append(run)
                df_rows['Step'].append(step)
                df_rows['Index'].append(index)
                df_rows[measure_name].append(measure_point)
    return pd.DataFrame(df_rows)

def winrate_estimation2df(run2winrate_estim):
    df_rows = {'Run': [], 'Index': [], 'Gamma':[], 'Winrate': []}
    for run, winrate_estim in run2winrate_estim.items():
        for index, array in enumerate(winrate_estim):
            for i in range(array.shape[0]):
                df_rows['Run'].append(run)
                df_rows['Index'].append(index)
                df_rows['Gamma'].append(array[i,0])
                df_rows['Winrate'].append(array[i,1])
    return pd.DataFrame(df_rows)

def plot_measure_per_agent(run2agent2measure, measure_name, log_y=False, yrange=None, optimal=None):
    # Generate DataFrame for Seaborn
    if type(run2agent2measure) != pd.DataFrame:
        df = measure_per_agent2df(run2agent2measure, measure_name)
    else:
        df = run2agent2measure
    df = df.reset_index()
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
            # plt.ylim(min_measure * factor, max_measure * 1.1)
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
    # Generate DataFrame for Seaborn
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
        # plt.ylim(min_measure * factor, max_measure * 1.1)
    else:
        plt.ylim(yrange[0], yrange[1])
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
    return df

def plot_winrate_estimation(run2winrate_estim):
    # Generate DataFrame for Seaborn
    if type(run2winrate_estim) != pd.DataFrame:
        df = winrate_estimation2df(run2winrate_estim)
    else:
        df = run2winrate_estim

    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title('Winrate Estimation', fontsize=FONTSIZE + 2)
    sns.lineplot(data=df, x="Gamma", y="Winrate", hue="Run", style="Index", ax=axes)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.ylabel('Winrate', fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.legend(loc='lower right', fontsize=FONTSIZE-4, ncol=3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/winrate_estimation.png", bbox_inches='tight')
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
    # Generate DataFrame for Seaborn
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
    # plt.show()
    return df
                

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--num_auctions', type=int, default=1, help='number of auction')

    args = parser.parse_args()

    with open('config/training.json') as f:
        training_config = json.load(f)


    # Set up Random Number Generator
    rng = np.random.default_rng(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])

    num_runs = training_config['num_runs']
    num_iter  = training_config['num_iter']
    record_interval = training_config['record_interval']
    update_interval = training_config['update_interval']

    # Max. number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    context_dim = training_config['context_dim']
    item_feature_var = training_config['item_feature_var']
    obs_context_dim = training_config['obs_context_dim']
    context_dist = training_config['context_distribution']
    random_bidding = training_config['random_bidding']

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse configuration file
    agent_configs, output_dir = parse_agent_config(args.config)
    output_dir = output_dir + time.strftime('%y%m%d-%H%M%S')
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

    run2agent2CTR = {}
    run2agent2CTR_RMSE = {}
    run2agent2CTR_bias = {}
    run2agent2winning_prob = {}

    run2auction_revenue = {}

    
    run2bidding = {}
    run2uncertainty = {}

    run2winrate_estimation = {}

    run2regret = {}
    run2optimal_selection_rate = {}
    run2bidding_error = {}
    run2winrate_optimal = {}
    run2bidding_optimal = {}
    run2utility_optimal = {}
    run2optimistic_CTR_ratio = {}

    num_auctions = args.num_auctions

    # check: agents in different auction has different item features
    auction2run2agents2items, auction2run2agents2item_values = draw_features(num_auctions, rng, num_runs, context_dim, agent_configs)

    # Repeated runs
    for run in range(num_runs):
        multi_auctions = []
        agents = instantiate_agents(rng, agent_configs, obs_context_dim, update_interval, context_dist, random_bidding)
        for auc in range(num_auctions):
            agents2items = auction2run2agents2items[auc][run]
            agents2item_values = auction2run2agents2item_values[auc][run]
            auction = instantiate_auction(
                rng, training_config, agents2items, agents2item_values, agents, max_slots, context_dim, obs_context_dim, context_dist)
            multi_auctions.append(auction)
        
        # Placeholders for summary statistics per run
        auction2agent2net_utility = defaultdict(lambda: defaultdict(list))

        auction2agent2allocation_regret = defaultdict(lambda: defaultdict(list))
        auction2agent2overbid_regret = defaultdict(lambda: defaultdict(list))
        auction2agent2underbid_regret = defaultdict(lambda: defaultdict(list))
        auction2agent2best_expected_value = defaultdict(lambda: defaultdict(list))

        auction2agent2CTR = defaultdict(lambda: defaultdict(list))
        auction2agent2CTR_RMSE = defaultdict(lambda: defaultdict(list))
        auction2agent2CTR_bias = defaultdict(lambda: defaultdict(list))

        auction2agent2winning_prob = defaultdict(lambda: defaultdict(list))

        auction_revenue = [[] for _ in range(num_auctions)]
        bidding = [[] for _ in range(num_auctions)]
        uncertainty = [[] for _ in range(num_auctions)]
        winrate_estimation = [[] for _ in range(num_auctions)]
        regret = [[] for _ in range(num_auctions)]
        optimal_selection_rate = [[] for _ in range(num_auctions)]
        bidding_error = [[] for _ in range(num_auctions)]
        winrate_optimal = [[] for _ in range(num_auctions)]
        bidding_optimal = [[] for _ in range(num_auctions)]
        utility_optimal = [[] for _ in range(num_auctions)]
        optimistic_CTR_ratio = [[] for _ in range(num_auctions)]

        # Run simulation (with global parameters -- fine for the purposes of this script)
        simulation_run(run, multi_auctions)

        # Store
        run2agent2net_utility[run] = auction2agent2net_utility
        run2agent2allocation_regret[run] = auction2agent2allocation_regret
        run2agent2overbid_regret[run] = auction2agent2overbid_regret
        run2agent2underbid_regret[run] = auction2agent2underbid_regret
        run2agent2best_expected_value[run] = auction2agent2best_expected_value

        run2agent2CTR[run] = auction2agent2CTR
        run2agent2CTR_RMSE[run] = auction2agent2CTR_RMSE
        run2agent2CTR_bias[run] = auction2agent2CTR_bias
        run2bidding[run] = bidding

        run2uncertainty[run] = uncertainty
        run2agent2winning_prob[run] = auction2agent2winning_prob

        run2winrate_estimation[run] = winrate_estimation

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
    plot_measure(run2optimal_selection_rate, 'Optimal Selection Rate')
    
    utility_df = measure_per_agent2df(run2agent2net_utility, 'Utility')
    optimal_utility_df = measure2df(run2utility_optimal, 'Utility')
    optimal_utility_df['Agent'] = 'Optimal'
    utility_df = pd.concat([utility_df, optimal_utility_df]).sort_values(['Agent', 'Run', 'Step'])
    utility_df['Utility (Cumulative)'] = utility_df.groupby(['Agent', 'Run'])['Utility'].cumsum()
    plot_measure_per_agent(utility_df, 'Utility')

    plot_measure_per_agent(utility_df, 'Utility (Cumulative)')
    utility_df.to_csv(f'{output_dir}/net_utility.csv', index=False)

    plot_measure_per_agent(run2agent2best_expected_value, '')

    allocation_regret_df = plot_measure_per_agent(run2agent2allocation_regret, 'Allocation Regret')
    allocation_regret_df.to_csv(f'{output_dir}/allocation_regret.csv', index=False)
    overbid_regret_df = plot_measure_per_agent(run2agent2overbid_regret, 'Overbid Regret')
    overbid_regret_df.to_csv(f'{output_dir}/overbid_regret.csv', index=False)
    underbid_regret_df = plot_measure_per_agent(run2agent2underbid_regret, 'Underbid Regret')
    underbid_regret_df.to_csv(f'{output_dir}/underbid_regret.csv', index=False)

    plot_measure(run2auction_revenue, 'Auction Revenue')

    uncertainty_df = plot_vector_measure(run2uncertainty, 'Uncertainty in Parameters')
    # uncertainty_df.to_csv(f'{output_dir}/uncertainty.csv', index=False)

    winning_prob_df = measure_per_agent2df(run2agent2winning_prob, 'Probability of Winning')
    optimal_winning_prob_df = measure2df(run2winrate_optimal, 'Probability of Winning')
    optimal_winning_prob_df['Agent'] = 'Optimal'
    winning_prob_df = pd.concat([winning_prob_df, optimal_winning_prob_df])
    plot_measure_per_agent(winning_prob_df, 'Probability of Winning')
    winning_prob_df.to_csv(f'{output_dir}/winning_probability.csv', index=False)

    plot_vector_measure(run2agent2CTR, 'CTR Value')
    # CTR_df.to_csv(f'{output_dir}/CTR.csv', index=False)

    plot_measure_per_agent(run2agent2CTR_RMSE, 'CTR RMSE')
    CTR_df = plot_measure_per_agent(run2agent2CTR_bias, 'CTR Bias', optimal=1.0)
    CTR_df.rename(columns={'CTR Bias':'CTR'}, inplace=True)
    CTR_df = CTR_df[~CTR_df['Agent'].str.startswith('Competitor')]
    CTR_df.drop(columns=['Agent'])
    CTR_df['Expected or Optimistic'] = 'Expected CTR'
    optimistic_CTR_df = measure2df(run2optimistic_CTR_ratio, 'CTR')
    optimistic_CTR_df['Expected or Optimistic'] = 'Optimistic CTR'
    CTR_df = pd.concat([CTR_df, optimistic_CTR_df])
    CTR_df.to_csv(f'{output_dir}/CTR.csv', index=False)
    plot_measure(CTR_df, 'CTR', hue='Expected or Optimistic')
    
    bidding_df = plot_vector_measure(run2bidding, 'Bidding')
    # bidding_df.to_csv(f'{output_dir}/Bidding.csv', index=False)

    regret_df = measure2df(run2regret, f'Regret({record_interval}steps)')
    regret_df['Regret'] = regret_df.groupby(['Run'])[f'Regret({record_interval}steps)'].cumsum()
    regret_df.to_csv(f'{output_dir}/regret.csv', index=False)
    plot_measure(regret_df, 'Regret')

    # plot_winrate_estimation(run2winrate_estimation)

    # net_utility_df_overall = utility_df.groupby(['Run', 'Step'])['Net Utility'].sum().reset_index().rename(columns={'Net Utility': 'Social Surplus'})
    # plot_measure(net_utility_df_overall, 'Social Surplus')
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
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from BidderAllocation import *  #  LogisticTSAllocator, OracleAllocator


def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''


def parse_agent_config(rng, context_dim, obs_context_dim, item_feature_var, path):
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

    # First sample item catalog (so it is consistent over different configs with the same seed)
    # Agent : (item_embedding, item_value)
    agents2items = {
        agent_config['name']: rng.normal(0.0, item_feature_var, size=(agent_config['num_items'], context_dim))
        for agent_config in agent_configs
    }

    agents2item_values = {
        agent_config['name']: rng.lognormal(0.1, 0.2, agent_config['num_items'])
        for agent_config in agent_configs
    }

    return agent_configs, agents2items, agents2item_values, output_dir


def instantiate_agents(rng, agent_configs, agents2item_values, agents2items, obs_context_dim):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              num_items=agent_config['num_items'],
              item_values=agents2item_values[agent_config['name']],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              context_dim = obs_context_dim,
              memory=('inf' if 'memory' not in agent_config.keys() else agent_config['memory']))
        for agent_config in agent_configs
    ]

    for agent in agents:
        if isinstance(agent.allocator, OracleAllocator):
            agent.allocator.update_item_embeddings(agents2items[agent.name])

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


def simulation_run(run):
    for i in tqdm(np.arange(1, num_iter+1), desc=f'run {run}'):
        auction.simulate_opportunity()

        if i%record_interval==0:
            for agent_id, agent in enumerate(auction.agents):
                agent2net_utility[agent.name].append(agent.get_net_utility())
                agent2gross_utility[agent.name].append(agent.get_gross_utility())

                agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
                agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
                agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
                agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

                agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
                agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

                agent2bidding_var[agent.name].append(agent.get_bidding_var())
                agent2uncertainty[agent.name].append(agent.get_uncertainty())

                if not isinstance(agent.bidder, TruthfulBidder):
                    agent2gamma[agent.name].append(np.array(agent.bidder.gammas))

                best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
                agent2best_expected_value[agent.name].append(best_expected_value)

                print('Average Best Value for Agent: ', best_expected_value)
                agent.move_index()

            auction_revenue.append(auction.revenue)
            auction.clear_revenue()

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

    num_runs = training_config['num_runs']
    num_iter  = training_config['num_iter']
    record_interval = training_config['record_interval']

    # Max. number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    context_dim = training_config['context_dim']
    item_feature_var = training_config['item_feature_var']
    obs_context_dim = training_config['obs_context_dim']
    context_dist = training_config['context_distribution']

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse configuration file
    agent_configs, agents2items, agents2item_values, output_dir = parse_agent_config(
        rng, context_dim, obs_context_dim, item_feature_var, args.config)

    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2agent2allocation_regret = {}
    run2agent2estimation_regret = {}
    run2agent2overbid_regret = {}
    run2agent2underbid_regret = {}
    run2agent2best_expected_value = {}

    run2agent2CTR_RMSE = {}
    run2agent2CTR_bias = {}
    run2agent2gamma = {}

    run2agent2bidding_var = {}
    run2agent2uncertainty = {}

    run2auction_revenue = {}

    # Repeated runs
    for run in range(num_runs):
        # Reinstantiate agents and auction per run
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items, obs_context_dim)
        auction  = instantiate_auction(
            rng, training_config, agents2items, agents2item_values, agents, max_slots, context_dim, obs_context_dim, context_dist)
        
        # Placeholders for summary statistics per run
        agent2net_utility = defaultdict(list)
        agent2gross_utility = defaultdict(list)
        agent2allocation_regret = defaultdict(list)
        agent2estimation_regret = defaultdict(list)
        agent2overbid_regret = defaultdict(list)
        agent2underbid_regret = defaultdict(list)
        agent2best_expected_value = defaultdict(list)

        agent2CTR_RMSE = defaultdict(list)
        agent2CTR_bias = defaultdict(list)
        agent2gamma = defaultdict(list)

        agent2bidding_var = defaultdict(list)
        agent2uncertainty = defaultdict(list)

        auction_revenue = []

        # Run simulation (with global parameters -- fine for the purposes of this script)
        simulation_run(run)

        # Store
        run2agent2net_utility[run] = agent2net_utility
        run2agent2gross_utility[run] = agent2gross_utility
        run2agent2allocation_regret[run] = agent2allocation_regret
        run2agent2estimation_regret[run] = agent2estimation_regret
        run2agent2overbid_regret[run] = agent2overbid_regret
        run2agent2underbid_regret[run] = agent2underbid_regret
        run2agent2best_expected_value[run] = agent2best_expected_value

        run2agent2CTR_RMSE[run] = agent2CTR_RMSE
        run2agent2CTR_bias[run] = agent2CTR_bias
        run2agent2gamma[run] = agent2gamma

        run2agent2bidding_var[run] = agent2bidding_var
        run2agent2uncertainty[run] = agent2uncertainty

        run2auction_revenue[run] = auction_revenue

    output_dir = output_dir + time.strftime('%y%m%d-%H%M%S')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.copy(args.config, os.path.join(output_dir, 'config.json'))

    def measure_per_agent2df(run2agent2measure, measure_name):
        df_rows = {'Run': [], 'Agent': [], 'Step': [], measure_name: []}
        for run, agent2measure in run2agent2measure.items():
            for agent, measures in agent2measure.items():
                for iteration, measure in enumerate(measures):
                    df_rows['Run'].append(run)
                    df_rows['Agent'].append(agent)
                    df_rows['Step'].append(iteration)
                    df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)
    
    def vector_of_measure_per_agent2df(run2agent2vector_measure, measure_name):
        df_rows = {'Run': [], 'Agent': [], 'Step': [], 'Parameter':[], measure_name: []}
        for run, agent2measure in run2agent2vector_measure.items():
            for agent, vectors in agent2measure.items():
                for iteration, vector in enumerate(vectors):
                    for index, measure in enumerate(vector):
                        df_rows['Run'].append(run)
                        df_rows['Agent'].append(agent)
                        df_rows['Step'].append(iteration)
                        df_rows['Parameter'].append(index)
                        df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_per_agent(run2agent2measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2measure) != pd.DataFrame:
            df = measure_per_agent2df(run2agent2measure, measure_name)
        else:
            df = run2agent2measure

        try:
            fig, axes = plt.subplots(figsize=FIGSIZE)
            plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
            min_measure, max_measure = 0.0, 0.0
            sns.lineplot(data=df, x="Step", y=measure_name, hue="Agent", ax=axes)
            plt.xticks(fontsize=FONTSIZE - 2)
            plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
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
            plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
        except Exception as e:
            print('cannot plot', e)
        return df
    
    def plot_vector_measure_per_agent(run2agent2vector_measure, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        # Generate DataFrame for Seaborn
        if type(run2agent2vector_measure) != pd.DataFrame:
            df = vector_of_measure_per_agent2df(run2agent2vector_measure, measure_name)
        else:
            df = run2agent2vector_measure

        df = df[~df['Agent'].str.startswith('Competitor')]

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Distribution', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.boxplot(data=df, x="Step", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel('Step', fontsize=FONTSIZE)
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
        plt.legend(loc='lower right', fontsize=FONTSIZE, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
        # plt.show()
        return df

    net_utility_df = plot_measure_per_agent(run2agent2net_utility, 'Net Utility').sort_values(['Agent', 'Run', 'Step'])
    net_utility_df.to_csv(f'{output_dir}/net_utility.csv', index=False)

    net_utility_df['Net Utility (Cumulative)'] = net_utility_df.groupby(['Agent', 'Run'])['Net Utility'].cumsum()
    plot_measure_per_agent(net_utility_df, 'Net Utility (Cumulative)')

    gross_utility_df = plot_measure_per_agent(run2agent2gross_utility, 'Gross Utility').sort_values(['Agent', 'Run', 'Step'])
    gross_utility_df.to_csv(f'{output_dir}/gross_utility.csv', index=False)

    gross_utility_df['Gross Utility (Cumulative)'] = gross_utility_df.groupby(['Agent', 'Run'])['Gross Utility'].cumsum()
    plot_measure_per_agent(gross_utility_df, 'Gross Utility (Cumulative)')

    plot_measure_per_agent(run2agent2best_expected_value, 'Mean Expected Value for Top Ad')

    plot_measure_per_agent(run2agent2allocation_regret, 'Allocation Regret')
    plot_measure_per_agent(run2agent2estimation_regret, 'Estimation Regret')
    overbid_regret_df = plot_measure_per_agent(run2agent2overbid_regret, 'Overbid Regret')
    overbid_regret_df.to_csv(f'{output_dir}/overbid_regret.csv', index=False)
    underbid_regret_df = plot_measure_per_agent(run2agent2underbid_regret, 'Underbid Regret')
    underbid_regret_df.to_csv(f'{output_dir}/underbid_regret.csv', index=False)

    plot_measure_per_agent(run2agent2CTR_RMSE, 'CTR RMSE', log_y=True)
    plot_measure_per_agent(run2agent2CTR_bias, 'CTR Bias', optimal=1.0) #, yrange=(.5, 5.0))

    bidding_var_df = plot_measure_per_agent(run2agent2bidding_var, 'Variance of Policy')
    uncertainty_df = plot_vector_measure_per_agent(run2agent2uncertainty, 'Uncertainty in Parameters')
    bidding_var_df.to_csv(f'{output_dir}/bidding_variance.csv', index=False)
    uncertainty_df.to_csv(f'{output_dir}/uncertainty.csv', index=False)
    
    shading_factor_df = plot_vector_measure_per_agent(run2agent2gamma, 'Shading Factors')

    def measure2df(run2measure, measure_name):
        df_rows = {'Run': [], 'Step': [], measure_name: []}
        for run, measures in run2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Step'].append(iteration)
                df_rows[measure_name].append(measure)
        return pd.DataFrame(df_rows)

    def plot_measure_overall(run2measure, measure_name):
        # Generate DataFrame for Seaborn
        if type(run2measure) != pd.DataFrame:
            df = measure2df(run2measure, measure_name)
        else:
            df = run2measure
        if 'Agent' in df.columns:
            df = df[~df['Agent'].str.startswith('Competitor')]

        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Time', fontsize=FONTSIZE + 2)
        sns.lineplot(data=df, x="Step", y=measure_name, ax=axes)
        min_measure = min(0.0, np.min(df[measure_name]))
        max_measure = max(0.0, np.max(df[measure_name]))
        plt.xlabel('Step', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        factor = 1.1 if min_measure < 0 else 0.9
        plt.ylim(min_measure * factor, max_measure * 1.1)
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
        # plt.show()
        return df

    auction_revenue_df = plot_measure_overall(run2auction_revenue, 'Auction Revenue')

    net_utility_df_overall = net_utility_df.groupby(['Run', 'Step'])['Net Utility'].sum().reset_index().rename(columns={'Net Utility': 'Social Surplus'})
    plot_measure_overall(net_utility_df_overall, 'Social Surplus')

    gross_utility_df_overall = gross_utility_df.groupby(['Run', 'Step'])['Gross Utility'].sum().reset_index().rename(columns={'Gross Utility': 'Social Welfare'})
    plot_measure_overall(gross_utility_df_overall, 'Social Welfare')

    auction_revenue_df['Measure Name'] = 'Auction Revenue'
    net_utility_df_overall['Measure Name'] = 'Social Surplus'
    gross_utility_df_overall['Measure Name'] = 'Social Welfare'

    columns = ['Run', 'Step', 'Measure', 'Measure Name']
    auction_revenue_df.columns = columns
    net_utility_df_overall.columns = columns
    gross_utility_df_overall.columns = columns

    pd.concat((auction_revenue_df, net_utility_df_overall, gross_utility_df_overall)).to_csv(f'{output_dir}/results.csv', index=False)

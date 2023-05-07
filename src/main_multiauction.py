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
        if 'num_copies' in agent_config.keys() and agent_config['num_copies']>1:
            for i in range(1, agent_config['num_copies'] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy['name'] += f' {i}'
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    return agent_configs, output_dir

def draw_features(num_auctions, rng, num_runs, context_dim, feature_dim, agent_configs):
    run2auction2items = {}
    run2auction2item_values = {}
    run2competitors2items = {}
    run2competitors2item_values = {}
    run2bilinear_map = {}

    for run in range(num_runs):
        auction2items = []
        auction2item_values = []
        competitor2items = {}
        competitor2item_values = {}

        for agent_config in agent_configs:
            if "Competitor" in agent_config['name']:
                temp = []
                for k in range(agent_config['num_items']):
                    feature = rng.normal(0.0, 1.0, size=feature_dim)
                    temp.append(feature)
                competitor2items[agent_config['name']] = np.stack(temp)
                competitor2item_values[agent_config['name']] = np.ones((agent_config['num_items'],))
            else:
                target_agent = agent_config
            
        for auction in range(num_auctions):
            temp = []
            for k in range(target_agent['num_items']):
                feature = rng.normal(0.0, 1.0, size=feature_dim)
                temp.append(feature)
            auction2items.append(np.stack(temp))
            auction2item_values.append(np.ones((target_agent['num_items'],)))

        run2auction2items[run] = auction2items
        run2auction2item_values[run] = auction2item_values
        run2competitors2items[run] = competitor2items
        run2competitors2item_values[run] = competitor2item_values
        run2bilinear_map[run] = rng.normal(0.0, 1.0, size=(context_dim, feature_dim))

    return run2auction2items, run2auction2item_values, run2competitors2items, run2competitors2item_values, run2bilinear_map


def instantiate_agents(rng, agent_configs, num_auctions, auction2items, auction2item_values,
                       competitors2items, competitors2item_values, obs_context_dim, update_interval, bilinear_map, random_bidding,
                       bonus_factors, optimism_scales, eq_winning_rates, overbidding_factors, overbidding_steps):
    # set up agents acting in different auctions
    # competitors, which determines the probability of winning, are shared across auctions
    competitors = []
    for agent_config in agent_configs:
        if 'Competitor' in agent_config['name']:
            competitors.append(agent_config)
        else:
            target_agent = agent_config
    
    # number of auctions must match the number of combinations of agent configurations
    assert len(bonus_factors) * len(optimism_scales) * len(eq_winning_rates) * len(overbidding_factors) * len(overbidding_steps) == num_auctions

    agents = []
    i = 0
    for bonus_factor in bonus_factors:
        for optimism_scale in optimism_scales:
            for eq_winning_rate in eq_winning_rates:
                for overbidding_factor in overbidding_factors:
                    for overbidding_step in overbidding_steps:
                        allocator = eval(f"{target_agent['allocator']['type']}(rng=rng, item_features=auction2items[i]{parse_kwargs(target_agent['allocator']['kwargs'])})")
                        bidder = eval(f"{target_agent['bidder']['type']}(rng=rng{parse_kwargs(target_agent['bidder']['kwargs'])})")
                        bidder.optimism_scale = optimism_scale
                        bidder.eq_winning_rate = eq_winning_rate
                        bidder.overbidding_factor = overbidding_factor
                        bidder.overbidding_steps = overbidding_step
                        agents.append(Agent(rng=rng,
                                            name=target_agent['name']+f" {i+1}",
                                            item_features=auction2items[i],
                                            item_values=auction2item_values[i],
                                            allocator=allocator,
                                            bidder=bidder,
                                            context_dim = obs_context_dim,
                                            update_interval=update_interval,
                                            random_bidding = random_bidding,
                                            memory=('inf' if 'memory' not in target_agent.keys() else target_agent['memory']),
                                            bonus_factor=bonus_factor))
                        i += 1
        
    competitors = [
        Agent(rng=rng,
              name=competitor['name'],
              item_features=competitors2items[competitor['name']],
              item_values=competitors2item_values[competitor['name']],
              allocator=eval(f"{competitor['allocator']['type']}(rng=rng, item_features=competitors2items[competitor['name']]{parse_kwargs(competitor['allocator']['kwargs'])})"),
              bidder=eval(f"{competitor['bidder']['type']}(rng=rng{parse_kwargs(competitor['bidder']['kwargs'])})"),
              context_dim = obs_context_dim,
              update_interval=update_interval,
              random_bidding = random_bidding,
              memory=('inf' if 'memory' not in competitor.keys() else competitor['memory']))
        for i, competitor in enumerate(competitors)
    ]

    for i, competitor in enumerate(competitors):
        if isinstance(competitor.allocator, OracleAllocator):
            competitor.allocator.set_CTR_model(bilinear_map)
        if isinstance(competitor.allocator, NeuralAllocator):
            competitor.allocator.initialize(competitors2item_values[competitor['name']])
        try:
            competitor.bidder.initialize(competitors2item_values[competitor['name']])
        except:
            pass
    
    for i, agent in enumerate(agents):
        if isinstance(agent.allocator, OracleAllocator):
            agent.allocator.set_CTR_model(bilinear_map)
        if isinstance(agent.allocator, NeuralAllocator):
            if i==0:
                agent.allocator.initialize(auctions2item_values[i])
            else:
                agent.allocator.copy_param(agents[0].allocator)
        try:
            if i==0:
                agent.bidder.initialize(auctions2item_values[i])
            else:
                agent.bidder.copy_param(agents[0].bidder)
        except:
            pass

    return agents, competitors


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
                    training_config['num_participants_per_round'],
                    enable_update=False)


def simulation_run(run, auctions):
    for i in tqdm(np.arange(1, int(num_iter/len(auctions))+1), desc=f'run {run}'):
        for j in range(len(auctions)):
            auctions[j].simulate_opportunity()
        
        if i % int(update_interval/len(auctions)) == 0:
            logs = []
            item_feature = []
            for agent, auction in zip(agents, auctions):
                logs.extend(agent.logs)
                ind = np.array(list(opp.item for opp in agent.logs))
                item_feature.append(agent.items[ind])
            item_feature = np.concatenate(item_feature, axis=0)

            contexts = np.array(list(opp.context for opp in logs))
            values = np.array(list(opp.value for opp in logs))
            bids = np.array(list(opp.bid for opp in logs))
            prices = np.array(list(opp.price for opp in logs))
            outcomes = np.array(list(opp.outcome for opp in logs))
            estimated_CTRs = np.array(list(opp.estimated_CTR for opp in logs))
            utilities = np.array(list(opp.utility for opp in logs))
            won_mask = np.array(list(opp.won for opp in logs))

            agents[0].allocator.update_(contexts[won_mask], item_feature[won_mask], outcomes[won_mask])
            agents[0].bidder.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, agents[0].name)
            for j in np.arange(1, num_auctions):
                agents[j].allocator.copy_param(agents[0].allocator)
                agents[j].bidder.copy_param(agents[0].bidder)
                agents[j].truncate_memory()
            
        if i % int(record_interval/len(auctions)) == 0:
            net_utility_ = 0
            allocation_regret_ = 0
            overbid_regret_ = 0
            underbid_regret_ = 0
            CTR_RMSE_ = 0
            CTR_bias_ = 0
            winning_prob_ = 0
            uncertainty_ = []
            optimal_selection_rate_ = 0
            bidding_error_ = []
            optimistic_CTR_ratio_ = 0
            regret_ = 0
            auction_revenue_ = 0
            winrate_optimal_ = 0
            utility_optimal_ = 0

            for agent, auction in zip(agents, auctions):
                net_utility_ += agent.get_net_utility()
                allocation_regret_ += agent.get_allocation_regret()
                overbid_regret_ += agent.get_overbid_regret()
                underbid_regret_ += agent.get_underbid_regret()
                CTR_RMSE_ += agent.get_CTR_RMSE()
                CTR_bias_ += agent.get_CTR_bias()
                winning_prob_ += agent.get_winning_prob()
                uncertainty_.append(agent.get_uncertainty())
                optimal_selection_rate_ += agent.get_optimal_selection_rate()
                bidding_error_.append(agent.get_bidding_error())
                optimistic_CTR_ratio_ += agent.get_optimistic_CTR_ratio()
                regret_ += auction.get_regret()
                auction_revenue_ += auction.revenue
                winrate_optimal_ += auction.get_winrate_optimal()
                utility_optimal_ += auction.get_optimal_utility()
            
            net_utility.append(net_utility_)
            allocation_regret.append(allocation_regret_)
            overbid_regret.append(overbid_regret_)
            underbid_regret.append(underbid_regret_)
            CTR_RMSE.append(CTR_RMSE_/len(auctions))
            CTR_bias.append(CTR_bias_/len(auctions))
            winning_prob.append(winning_prob_/len(auctions))
            uncertainty.append(np.concatenate(uncertainty_))
            optimal_selection_rate.append(optimal_selection_rate_/len(auctions))
            bidding_error.append(np.concatenate(bidding_error_))
            optimistic_CTR_ratio.append(optimistic_CTR_ratio_/len(auctions))
            regret.append(regret_)
            auction_revenue.append(auction_revenue_)
            winrate_optimal.append(winrate_optimal_/len(auctions))
            utility_optimal.append(utility_optimal_)


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


def plot_measure(run2measure, measure_name, hue=None, optimal=None):
    # Generate DataFrame for Seaborn
    if type(run2measure) != pd.DataFrame:
        df = measure2df(run2measure, measure_name)
    else:
        df = run2measure
    df = df.reset_index()

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
    if optimal is not None:
        plt.axhline(optimal, ls='--', color='gray', label='Optimal')
        min_measure = min(min_measure, optimal)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}.png", bbox_inches='tight')
    # plt.show()
    return df

def measure2df_auc(run2measure, measure_name):
    df_rows = {'Run': [], 'Auction': [], 'Step': [], measure_name: []}
    for run, measures in run2measure.items():
        for i, auction in enumerate(measures):
            for iteration, measure in enumerate(auction):
                df_rows['Run'].append(run)
                df_rows['Auction'].append(i)
                df_rows['Step'].append(iteration)
                df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)

def plot_measure_auc(run2measure, measure_name, hue=None):
    # Generate DataFrame for Seaborn
    if type(run2measure) != pd.DataFrame:
        df = measure2df_auc(run2measure, measure_name)
    else:
        df = run2measure
    df = df.reset_index()

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

    args = parser.parse_args()

    with open('config/training_multiauction.json') as f:
        training_config = json.load(f)        

    # Set up Random Number Generator
    rng = np.random.default_rng(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])

    num_auctions = training_config['num_auctions']
    num_runs = training_config['num_runs']
    num_iter  = training_config['num_iter']
    record_interval = training_config['record_interval']
    update_interval = training_config['update_interval']

    # Technical parameters for distribution of latent embeddings
    max_slots = 1
    context_dim = training_config['context_dim']
    obs_context_dim = training_config['obs_context_dim']
    context_dist = training_config['context_distribution']
    random_bidding = training_config['random_bidding']
    feature_dim = training_config['feature_dim']

    bonus_factors = training_config['bonus_factor']
    optimism_scales = training_config['optimism_scale']
    eq_winning_rates = training_config['eq_winning_rate']
    overbidding_factors = training_config['overbidding_factor']
    overbidding_steps = training_config['overbidding_steps']
    overbidding_steps = [int(steps/num_auctions) for steps in overbidding_steps]

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse configuration file
    agent_configs, output_dir = parse_agent_config(args.config)
    output_dir = output_dir + time.strftime('%y%m%d-%H%M%S')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.copy(args.config, os.path.join(output_dir, 'agent_config.json'))
    shutil.copy('config/training_multiauction.json', os.path.join(output_dir, 'training_config.json'))

    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2net_utility = {}
    run2allocation_regret = {}
    run2overbid_regret = {}
    run2underbid_regret = {}
    run2best_expected_value = {}

    run2CTR_RMSE = {}
    run2CTR_bias = {}
    run2winning_prob = {}

    run2auction_revenue = {}
    
    run2uncertainty = {}

    run2regret = {}
    run2optimal_selection_rate = {}
    run2bidding_error = {}
    run2winrate_optimal = {}
    run2utility_optimal = {}
    run2optimistic_CTR_ratio = {}

    # check: agents in different auction has different item features
    run2auction2items, run2auction2item_values, run2competitors2items, run2competitors2item_values, run2bilinear_map = \
    draw_features(num_auctions, rng, num_runs, context_dim, feature_dim, agent_configs)

    # Repeated runs
    for run in range(num_runs):
        bilinear_map = run2bilinear_map[run]
        auctions2items = run2auction2items[run]
        auctions2item_values = run2auction2item_values[run]
        agents, competitors = instantiate_agents(rng, agent_configs, num_auctions, run2auction2items[run], run2auction2item_values[run],
                       run2competitors2items[run], run2competitors2item_values[run], obs_context_dim, update_interval, bilinear_map, random_bidding,
                       bonus_factors, optimism_scales, eq_winning_rates, overbidding_factors, overbidding_steps)
        
        auctions = []
        for i in range(num_auctions):
            agents_for_auction = [agents[i], *competitors]
            items_for_auction = {competitor.name : competitor.items for competitor in competitors}
            items_for_auction[agents[i].name] = agents[i].items
            values_for_auction = {competitor.name : competitor.item_values for competitor in competitors}
            values_for_auction[agents[i].name] = agents[i].item_values
            auction = instantiate_auction(
                rng, training_config, bilinear_map, items_for_auction, values_for_auction, agents_for_auction, max_slots, context_dim, obs_context_dim, context_dist)
            auctions.append(auction)
        
        net_utility = []
        allocation_regret = []
        overbid_regret = []
        underbid_regret = []
        best_expected_value = []

        CTR_RMSE = []
        CTR_bias = []

        winning_prob = []

        uncertainty = []
        optimal_selection_rate = []
        bidding_error = []
        optimistic_CTR_ratio = []

        winrate_estimation = []
        regret = []
        auction_revenue = []
        winrate_optimal = []
        utility_optimal = []

        # Run simulation (with global parameters -- fine for the purposes of this script)
        simulation_run(run, auctions)

        run2net_utility[run] = net_utility
        run2allocation_regret[run] = allocation_regret
        run2overbid_regret[run] = overbid_regret
        run2underbid_regret[run] = underbid_regret
        run2best_expected_value[run] = best_expected_value

        run2CTR_RMSE[run] = CTR_RMSE
        run2CTR_bias[run] = CTR_bias

        run2uncertainty[run] = uncertainty
        run2winning_prob[run] = winning_prob

        run2auction_revenue[run] = auction_revenue
        run2regret[run] = regret
        run2optimal_selection_rate[run] = optimal_selection_rate
        run2bidding_error[run] = bidding_error
        run2winrate_optimal[run] = winrate_optimal
        run2utility_optimal[run] = utility_optimal
        run2optimistic_CTR_ratio[run] = optimistic_CTR_ratio


    optimal_df=plot_measure(run2optimal_selection_rate, 'Optimal Selection Rate')
    optimal_df.to_csv(f'{output_dir}/optimal_df.csv', index=False)

    utility_df = measure2df(run2net_utility, 'Utility')
    utility_df['Agent'] = 'Agent'
    optimal_utility_df = measure2df(run2utility_optimal, 'Utility')
    optimal_utility_df['Agent'] = 'Optimal'
    utility_df = pd.concat([utility_df, optimal_utility_df]).sort_values(['Agent', 'Run', 'Step'])
    utility_df['Utility (Cumulative)'] = utility_df.groupby(['Agent', 'Run'])['Utility'].cumsum()
    plot_measure_per_agent(utility_df, 'Utility')
    plot_measure_per_agent(utility_df, 'Utility (Cumulative)')
    utility_df.to_csv(f'{output_dir}/net_utility.csv', index=False)

    allocation_regret_df = plot_measure(run2allocation_regret, 'Allocation Regret')
    allocation_regret_df.to_csv(f'{output_dir}/allocation_regret.csv', index=False)
    overbid_regret_df = plot_measure(run2overbid_regret, 'Overbid Regret')
    overbid_regret_df.to_csv(f'{output_dir}/overbid_regret.csv', index=False)
    underbid_regret_df = plot_measure(run2underbid_regret, 'Underbid Regret')
    underbid_regret_df.to_csv(f'{output_dir}/underbid_regret.csv', index=False)
    plot_measure(run2auction_revenue, 'Auction Revenue')

    winning_prob_df = measure2df(run2winning_prob, 'Probability of Winning')
    winning_prob_df['Agent'] = 'Agent'
    optimal_winning_prob_df = measure2df(run2winrate_optimal, 'Probability of Winning')
    optimal_winning_prob_df['Agent'] = 'Optimal'
    winning_prob_df = pd.concat([winning_prob_df, optimal_winning_prob_df])
    plot_measure_per_agent(winning_prob_df, 'Probability of Winning')
    winning_prob_df.to_csv(f'{output_dir}/winning_probability.csv', index=False)

    plot_measure(run2CTR_RMSE, 'CTR RMSE')
    CTR_df = plot_measure(run2CTR_bias, 'CTR Bias', optimal=1.0)
    CTR_df.rename(columns={'CTR Bias':'CTR'}, inplace=True)
    CTR_df['Expected or Optimistic'] = 'Expected CTR'
    optimistic_CTR_df = measure2df(run2optimistic_CTR_ratio, 'CTR')
    optimistic_CTR_df['Expected or Optimistic'] = 'Optimistic CTR'
    CTR_df = pd.concat([CTR_df, optimistic_CTR_df])
    CTR_df.to_csv(f'{output_dir}/CTR.csv', index=False)
    plot_measure(CTR_df, 'CTR', hue='Expected or Optimistic')

    regret_df = measure2df(run2regret, f'Regret({record_interval}steps)')
    regret_df['Regret'] = regret_df.groupby(['Run'])[f'Regret({record_interval}steps)'].cumsum()
    regret_df.to_csv(f'{output_dir}/regret.csv', index=False)
    plot_measure(regret_df, 'Regret')
    
    plot_vector_measure(run2bidding_error, 'Bidding Error')
    uncertainty_df = plot_vector_measure(run2uncertainty, 'Uncertainty in Parameters')
    # uncertainty_df.to_csv(f'{output_dir}/uncertainty.csv', index=False)
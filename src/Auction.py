from AuctionAllocation import AllocationMechanism
from Bidder import Bidder

import numpy as np
from scipy.stats import norm

from BidderAllocation import OracleAllocator
from Bidder import OracleBidder
from Models import sigmoid
from Impression import ImpressionOpportunity

class Auction:
    ''' Base class for auctions '''
    def __init__(self, rng, allocation, agents, bilinear_map, agent2items, agents2item_values, max_slots, context_dim, obs_context_dim, context_dist, num_participants_per_round):
        self.rng = rng
        self.allocation = allocation
        self.agents = agents
        self.max_slots = max_slots
        self.revenue = .0

        self.M = bilinear_map
        self.agent2items = agent2items
        self.agents2item_values = agents2item_values

        self.context_dim = context_dim
        self.obs_context_dim = obs_context_dim
        self.context_dist = context_dist # Gaussian, Bernoulli, Uniform
        self.gaussian_var = 1.0
        self.bernoulli_p = 0.5

        self.true_winrate = []
        self.regret = []
        self.optimal_bidding = []
        self.win_optimal = []
        self.optimal_utility = []

        self.num_participants_per_round = num_participants_per_round
    
    def generate_context(self):
        if self.context_dist=='Gaussian':
            return self.rng.normal(0.0, 1.0, size=self.context_dim)
        elif self.context_dist=='Bernoulli':
            return self.rng.binomial(1, self.bernoulli_p, size=self.context_dim)
        else:
            return self.rng.uniform(-1.0, 1.0, size=self.context_dim)

    def CTR(self, context, item_features):
        # CTR = sigmoid(context @ M @ item_feature) for each item
        return sigmoid(item_features @ self.M.T @ context / np.sqrt(self.context_dim*item_features.shape[1]))

    def simulate_opportunity(self):
        # Sample the number of slots uniformly between [1, max_slots]
        num_slots = self.rng.integers(1, self.max_slots + 1)

        # Sample a true context vector
        true_context = self.generate_context()

        # Mask true context into observable context
        obs_context = true_context[:self.obs_context_dim]

        # At this point, the auctioneer solicits bids from
        # the list of bidders that might want to compete.
        bids = []
        CTRs = []
        participating_agents_idx = self.rng.choice(len(self.agents), self.num_participants_per_round, replace=False)
        participating_agents = [self.agents[idx] for idx in participating_agents_idx]

        for agent in participating_agents:
            true_CTR = self.CTR(true_context, self.agent2items[agent.name])
            expected_value = self.agents2item_values[agent.name] * true_CTR
            best_value = np.max(expected_value)

            if isinstance(agent.allocator, OracleAllocator):
                bid, item = agent.bid(true_context)
            elif isinstance(agent.bidder, OracleBidder):
                item, estimated_CTR = agent.select_item(obs_context)
                value = agent.item_values[item]
                b_grid = np.linspace(0.1*value, 1.5*value, 200)
                prob_win = self.winrate_grid(participating_agents, true_context, b_grid)
                bid, item = agent.bid(obs_context, value, prob_win, b_grid)
                utility = prob_win * (np.max(expected_value) - b_grid)
                p = self.winrate_point(participating_agents, true_context, bid)
                self.regret.append(np.max(utility) - p*(expected_value[item] - bid))
            else:
                bid, item = agent.bid(obs_context)
            bids.append(bid)
            
            agent.logs[-1].set_true_CTR(best_value, true_CTR[item])
            CTRs.append(true_CTR[item])
            if not agent.name.startswith('Competitor') and not isinstance(agent.bidder, OracleBidder):
                regret, u_optimal, b_optimal = self.compute_regret(participating_agents, true_context, bid, expected_value, item)
                self.regret.append(regret)
                agent.logs[-1].bidding_error = bid - b_optimal
                self.optimal_bidding.append(b_optimal)
                self.win_optimal.append(self.winrate_point(participating_agents, true_context, b_optimal))
                self.optimal_utility.append(u_optimal)
        bids = np.array(bids)
        CTRs = np.array(CTRs)

        # Now we have bids, we need to somehow allocate slots
        # "second_prices" tell us how much lower the winner could have gone without changing the outcome
        winners, prices, second_prices = self.allocation.allocate(bids, num_slots)

        # Bidders only obtain value when they get their outcome
        # Either P(view), P(click | view, ad), P(conversion | click, view, ad)
        # For now, look at P(click | ad) * P(view)
        outcomes = self.rng.binomial(1, CTRs[winners])

        # Let bidders know what they're being charged for
        for slot_id, (winner, price, second_price, outcome) in enumerate(zip(winners, prices, second_prices, outcomes)):
            for agent_id, agent in enumerate(participating_agents):
                if agent_id == winner:
                    agent.charge(price, second_price, bool(outcome))
                else:
                    agent.set_price(price)
            self.revenue += price
        for agent in participating_agents:
            agent.update()

    def winrate_grid(self, agents, context, b_grid):
        p_grid = np.ones_like(b_grid)
        for agent in agents:
            if agent.name.startswith('Competitor'):
                CTR = agent.allocator.estimate_CTR(context)
                expected_value = CTR*self.agents2item_values[agent.name]
                mean = np.max(expected_value) * agent.bidder.bias
                ind = np.argmax(expected_value)
                std = agent.bidder.noise * self.agents2item_values[agent.name][ind]
                if std==0.0:
                    for (i), b in np.ndenumerate(b_grid):
                        p_grid[i] *= 0 if mean>b else 1
                else:
                    for (i), b in np.ndenumerate(b_grid):
                        p_grid[i] *= norm.cdf(b, loc=mean, scale=std)
        return p_grid
    
    def winrate_point(self, agents, context, b):
        p = 1.0
        for agent in agents:
            if agent.name.startswith('Competitor'):
                CTR = agent.allocator.estimate_CTR(context)
                mean = np.max(CTR*self.agents2item_values[agent.name]) * agent.bidder.bias
                std = agent.bidder.noise
                if std==0:
                    p *= 0 if mean>b else 1
                else:
                    p *= norm.cdf(b, loc=mean, scale=std)
        return p
    
    def compute_regret(self, agents, context, bid, expected_value, item):
        best_value = np.max(expected_value)
        b_grid = np.linspace(0.0, 1.0*best_value, 200)
        p_grid = self.winrate_grid(agents, context, b_grid)
        utility = p_grid * (best_value - b_grid)
        p = self.winrate_point(agents, context, bid)
        return np.max(utility) - p*(expected_value[item] - bid), np.max(utility), b_grid[np.argmax(utility)]

    def clear_revenue(self):
        self.revenue = 0.0
    
    def get_regret(self):
        for agent in self.agents:
            if not agent.name.startswith('Competitor'):
                index = agent.record_index
                break
        return np.sum(self.regret[index:])
    
    def get_optimal_bidding(self):
        for agent in self.agents:
            if not agent.name.startswith('Competitor'):
                index = agent.record_index
                break
        return np.array(self.regret[index:])

    def get_winrate_optimal(self):
        for agent in self.agents:
            if not agent.name.startswith('Competitor'):
                index = agent.record_index
                break
        return np.mean(self.win_optimal[index:])
    
    def get_optimal_utility(self):
        for agent in self.agents:
            if not agent.name.startswith('Competitor'):
                index = agent.record_index
                break
        return np.sum(self.optimal_utility[index:])
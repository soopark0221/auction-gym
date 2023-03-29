from AuctionAllocation import AllocationMechanism
from Bidder import Bidder

import numpy as np

from BidderAllocation import OracleAllocator
from Models import sigmoid

class Auction:
    ''' Base class for auctions '''
    def __init__(self, rng, allocation, agents, agent2items, agents2item_values, max_slots, context_dim, obs_context_dim, context_dist, num_participants_per_round):
        self.rng = rng
        self.allocation = allocation
        self.agents = agents
        self.max_slots = max_slots
        self.revenue = .0

        self.agent2items = agent2items
        self.agents2item_values = agents2item_values

        self.context_dim = context_dim
        self.obs_context_dim = obs_context_dim
        self.context_dist = context_dist # Gaussian, Bernoulli, Uniform
        self.gaussian_var = 1.0
        self.bernoulli_p = 0.5

        self.num_participants_per_round = num_participants_per_round
    
    def generate_context(self):
        if self.context_dist=='Gaussian':
            return self.rng.normal(0.0, 1.0, size=self.context_dim)
        elif self.context_dist=='Bernoulli':
            return self.rng.binomial(1, self.bernoulli_p, size=self.context_dim)
        else:
            return self.rng.uniform(-1.0, 1.0, size=self.context_dim)
    
    def CTR(self, context, item_features):
        return sigmoid(context @ item_features.T / np.sqrt(self.context_dim))

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
            # Get the bid and the allocated item
            if isinstance(agent.allocator, OracleAllocator):
                bid, item = agent.bid(true_context)
            else:
                bid, item = agent.bid(obs_context)
            bids.append(bid)
            # Compute the true CTRs for items in this agent's catalogue
            true_CTR = self.CTR(true_context, self.agent2items[agent.name])
            agent.logs[-1].set_true_CTR(np.max(true_CTR * self.agents2item_values[agent.name]), true_CTR[item])
            CTRs.append(true_CTR[item])
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

    def clear_revenue(self):
        self.revenue = 0.0

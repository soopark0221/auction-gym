import numpy as np
from scipy.stats import norm

from BidderAllocation import *
from Bidder import OracleBidder
from Impression import ImpressionOpportunity
from Models import sigmoid
from Bidder import TruthfulBidder


class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, item_features, item_values, allocator, bidder, context_dim, update_interval, bonus_factor=0.0, explore_then_commit=0):
        self.rng = rng
        self.name = name
        self.items = item_features
        self.item_values = item_values
        self.num_items = item_features.shape[0]
        self.feature_dim = item_features.shape[1]
        self.context_dim = context_dim

        self.logs = []

        self.allocator = allocator
        self.bidder = bidder
        
        self.clock = 1
        self.record_index = 0
        self.update_interval = update_interval
        self.explore_then_commit = explore_then_commit

        self.Gram = [np.zeros((self.context_dim, self.context_dim)) for _ in range(self.num_items)]
        self.bonus_factor = bonus_factor

    def select_item(self, context):
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context, UCB=True)
            bonus = np.array([context @ self.Gram[i] @ context for i in range(self.num_items)])/np.sum(context**2)
            bonus = np.sqrt(1/(bonus+1e-2))
            best_item = np.argmax(self.item_values * (estim_CTRs + self.allocator.c * uncertainty) + self.bonus_factor * bonus)

        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
            uncertainty = 0.0
            bonus = np.array([context @ self.Gram[i] @ context for i in range(self.num_items)])/np.sum(context**2)
            bonus = np.sqrt(1/(bonus+1e-2))
            best_item = np.argmax(self.item_values * estim_CTRs + self.bonus_factor * bonus)

        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()

        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            return best_item, estim_CTRs[best_item], uncertainty[best_item]
        else:
            return best_item, estim_CTRs[best_item], uncertainty

    def bid(self, context, value=None, prob_win=None, b_grid=None):
        best_item, estimated_CTR, uncertainty = self.select_item(context)
        optimistic_CTR = estimated_CTR

        value = self.item_values[best_item]

        if not isinstance(self.allocator, OracleAllocator):
            context = context[:self.context_dim]

        if isinstance(self.bidder, OracleBidder):
            bid = self.bidder.bid(value, estimated_CTR, prob_win, b_grid)
        elif not isinstance(self.allocator, OracleAllocator):
            bid, optimistic_CTR = self.bidder.bid(value, context, estimated_CTR, uncertainty)
        else:
            bid = self.bidder.bid(value, context, estimated_CTR)

        if not isinstance(self.allocator, OracleAllocator) and self.clock<self.explore_then_commit:
            bid = value *(1.0  + self.rng.normal(0.0, 1.0) * 0.02)
            propensity = norm.pdf(bid, loc=0.0, scale=0.02)
            if not isinstance(self.bidder, TruthfulBidder):
                self.bidder.b.append(bid)
                self.bidder.propensity.append(propensity)

        self.logs.append(ImpressionOpportunity(context=context,
                                               item=best_item,
                                               estimated_CTR=estimated_CTR,
                                               optimistic_CTR=optimistic_CTR,
                                               value=value,
                                               bid=bid,
                                               best_expected_value=0.0,
                                               true_CTR=0.0,
                                               price=0.0,
                                               second_price=0.0,
                                               outcome=0,
                                               won=False,
                                               utility=0.0,
                                               optimal_item=False,
                                               bidding_error=0.0))

        self.clock += 1
        self.Gram[best_item] += np.outer(context, context) / np.sum(context**2)
        return bid, best_item

    def charge(self, price, second_price, outcome):
        self.logs[-1].set_price_outcome(price, second_price, outcome, won=True)
        last_value = self.logs[-1].value * outcome
        self.logs[-1].utility += last_value - price

    def set_price(self, price):
        self.logs[-1].set_price(price)

    def update(self):
        if self.clock%self.update_interval or self.clock<self.explore_then_commit:
            return
        
        contexts = np.array(list(opp.context for opp in self.logs))
        items = np.array(list(opp.item for opp in self.logs))
        values = np.array(list(opp.value for opp in self.logs))
        bids = np.array(list(opp.bid for opp in self.logs))
        prices = np.array(list(opp.price for opp in self.logs))
        outcomes = np.array(list(opp.outcome for opp in self.logs))
        estimated_CTRs = np.array(list(opp.estimated_CTR for opp in self.logs))
        utilities = np.array(list(opp.utility for opp in self.logs))

        won_mask = np.array(list(opp.won for opp in self.logs))
        self.allocator.update(contexts[won_mask], items[won_mask], outcomes[won_mask], self.name)

        self.bidder.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, self.name)

    def get_allocation_regret(self):
        ''' How much value am I missing out on due to suboptimal allocation? '''
        return np.sum(list(opp.best_expected_value - opp.true_CTR * opp.value for opp in self.logs[self.record_index:]))

    def get_overbid_regret(self):
        ''' How much am I overpaying because I could shade more? '''
        return np.sum(list((opp.price - opp.second_price) * opp.won for opp in self.logs[self.record_index:]))

    def get_underbid_regret(self):
        ''' How much have I lost because I could have shaded less? '''
        # The difference between the winning price and our bid -- for opportunities we lost, and where we could have won without overpaying
        # Important to mention that this assumes a first-price auction! i.e. the price is the winning bid
        return np.sum(list((opp.price - opp.bid) * (not opp.won) * (opp.price < (opp.true_CTR * opp.value)) for opp in self.logs[self.record_index:]))

    def get_CTR_RMSE(self):
        return np.sqrt(np.mean(list((opp.true_CTR - opp.estimated_CTR)**2 for opp in self.logs[self.record_index:])))

    def get_CTR_bias(self):
        return np.mean(list((opp.estimated_CTR / opp.true_CTR) for opp in self.logs[self.record_index:]))
    
    def get_optimistic_CTR_ratio(self):
        return np.mean(list((opp.optimistic_CTR / opp.true_CTR) for opp in self.logs[self.record_index:]))
    
    def get_uncertainty(self):
        return self.allocator.get_uncertainty()
    
    def move_index(self):
        self.record_index = len(self.logs)

    def get_net_utility(self):
        return np.sum(list(opp.utility for opp in self.logs[self.record_index:]))

    def get_winning_prob(self):
        return np.mean(list(opp.won for opp in self.logs[self.record_index:]))

    def get_optimal_selection_rate(self):
        return np.mean(list(float(opp.optimal_item) for opp in self.logs[self.record_index:]))
    
    def get_bidding_error(self):
        return np.array(list(opp.bidding_error for opp in self.logs[self.record_index:]))
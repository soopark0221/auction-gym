import numpy as np

from BidderAllocation import *
from Bidder import OracleBidder
from Impression import ImpressionOpportunity
from Models import sigmoid
from Bidder import TruthfulBidder


class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, item_features, item_values, allocator, bidder, context_dim, update_interval, random_bidding, memory, bonus_factor=0.5):
        self.rng = rng
        self.name = name
        self.items = item_features
        self.num_items = item_features.shape[0]
        self.feature_dim = item_features.shape[1]
        self.context_dim = context_dim

        split = random_bidding.split()
        self.random_bidding_mode = split[0]    # none or uniform or gaussian noise
        if self.random_bidding_mode!='None':
            self.init_num_random_bidding = int(split[1])
            self.decay_factor = float(split[2])

        self.use_optimistic_value = True

        # Value distribution
        self.item_values = item_values

        self.logs = []

        self.allocator = allocator
        self.bidder = bidder
        
        self.clock = 1
        self.record_index = 0

        self.update_interval = update_interval
        self.memory = memory

        self.item_selection = np.zeros((self.num_items,)).astype(int)
        self.Gram = [np.zeros((self.context_dim, self.context_dim)) for _ in range(self.num_items)]
        self.bonus_factor = bonus_factor
    
    def should_explore(self):
        if isinstance(self.bidder, OracleBidder) or self.random_bidding_mode=='None':
            return False
        if (self.allocator.mode=='TS' or self.allocator.mode=='UCB') and self.use_optimistic_value:
            return self.clock%self.update_interval < \
            self.init_num_random_bidding/np.power(self.decay_factor, int(self.clock/self.update_interval))/4
        else:
            return self.clock%self.update_interval < \
                self.init_num_random_bidding/np.power(self.decay_factor, int(self.clock/self.update_interval))

    def select_item(self, context):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs, uncertainty = self.allocator.estimate_CTR(context, UCB=True)
            if isinstance(self.allocator, LogisticAllocatorM):
                best_item = np.argmax(self.item_values * (estim_CTRs + self.allocator.c * uncertainty))
            else:
                # bonus = np.sqrt(np.log(self.clock)/(self.item_selection + 1e-2))
                bonus = np.array([context @ self.Gram[i] @ context for i in range(self.num_items)])/np.sum(context**2)
                bonus = np.sqrt(1/(bonus+1e-2))
                best_item = np.argmax(self.item_values * (estim_CTRs + self.allocator.c * uncertainty) + self.bonus_factor * bonus)
                # best_item = np.argmax(self.item_values * (estim_CTRs + self.allocator.c * uncertainty))

        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs = self.allocator.estimate_CTR(context, TS=True)
            TS_CTR = []
            for i in range(5):
                TS_CTR.append(self.allocator.estimate_CTR(context, TS=True).reshape(-1))
            TS_CTR = np.stack(TS_CTR)
            if isinstance(self.allocator, LogisticAllocatorM):
                best_item = np.argmax(self.item_values * estim_CTRs)
            else:
                # bonus = np.sqrt(np.log(self.clock)/(self.item_selection + 1e-2))
                bonus = np.array([context @ self.Gram[i] @ context for i in range(self.num_items)])/np.sum(context**2)
                bonus = np.sqrt(1/(bonus+1e-2))
                best_item = np.argmax(self.item_values * estim_CTRs + self.bonus_factor * bonus)
                # best_item = np.argmax(self.item_values * estim_CTRs)
            estim_CTRs = self.allocator.estimate_CTR(context)
            uncertainty = np.std(TS_CTR, axis=0)

        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
            uncertainty = 0.0
            if isinstance(self.allocator, OracleAllocator) or isinstance(self.allocator, LogisticAllocatorM):
                best_item = np.argmax(self.item_values * estim_CTRs)
            else:
                # bonus = np.sqrt(np.log(self.clock)/(self.item_selection + 1e-2))
                bonus = np.array([context @ self.Gram[i] @ context for i in range(self.num_items)])/np.sum(context**2)
                bonus = np.sqrt(1/(bonus+1e-2))
                best_item = np.argmax(self.item_values * estim_CTRs + self.bonus_factor * bonus)
                # best_item = np.argmax(self.item_values * estim_CTRs)

        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()

        if not isinstance(self.allocator, OracleAllocator) and (self.allocator.mode=='UCB' or self.allocator.mode=='TS'):
            if self.clock < 0:
                best_item = self.rng.choice(self.num_items, 1).item()
            return best_item, estim_CTRs[best_item], uncertainty[best_item]
            
        else:
            return best_item, estim_CTRs[best_item], uncertainty

    def bid(self, context, value=None, prob_win=None, b_grid=None):
        # First, pick what item we want to choose
        best_item, estimated_CTR, uncertainty = self.select_item(context)
        optimistic_CTR = estimated_CTR

        # Sample value for this item
        value = self.item_values[best_item]

        if isinstance(self.allocator, OracleAllocator):
            context =context[:self.context_dim]

        if isinstance(self.bidder, OracleBidder):
            # todo: modify oraclebidder
            raise NotImplementedError
            bid = self.bidder.bid(value, estimated_CTR, prob_win, b_grid)
        elif not isinstance(self.allocator, OracleAllocator) and self.should_explore():
            if self.random_bidding_mode=='uniform':
                bid = self.rng.uniform(0, value*1.5)
            elif self.random_bidding_mode=='overbidding-uniform':
                bid = self.rng.uniform(value*1.0, value*1.5)
            elif self.random_bidding_mode=='gaussian':
                bid = self.bidder.bid(value, context, estimated_CTR)
                bid += value * self.rng.normal(0, 0.5)
                bid = np.maximum(bid, 0)
            if not isinstance(self.bidder, TruthfulBidder):
                self.bidder.b.append(bid)
        else:
            if not isinstance(self.allocator, OracleAllocator):
                bid, optimistic_CTR = self.bidder.bid(value, context, estimated_CTR, uncertainty)
            else:
                bid = self.bidder.bid(value, context, estimated_CTR)

        # Log what we know so far
        self.logs.append(ImpressionOpportunity(context=context,
                                               item=best_item,
                                               estimated_CTR=estimated_CTR,
                                               optimistic_CTR=optimistic_CTR,
                                               value=value,
                                               bid=bid,
                                               # These will be filled out later
                                               best_expected_value=0.0,
                                               true_CTR=0.0,
                                               price=0.0,
                                               second_price=0.0,
                                               outcome=0,
                                               won=False,
                                               utility=0.0,
                                               optimal_item=False,
                                               bidding_error=0.0))

        self.item_selection[best_item] += 1
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
        if self.clock%self.update_interval:
            return
        # Gather relevant logs
        contexts = np.array(list(opp.context for opp in self.logs))
        items = np.array(list(opp.item for opp in self.logs))
        values = np.array(list(opp.value for opp in self.logs))
        bids = np.array(list(opp.bid for opp in self.logs))
        prices = np.array(list(opp.price for opp in self.logs))
        outcomes = np.array(list(opp.outcome for opp in self.logs))
        estimated_CTRs = np.array(list(opp.estimated_CTR for opp in self.logs))
        utilities = np.array(list(opp.utility for opp in self.logs))

        # Update response model with data from winning bids
        won_mask = np.array(list(opp.won for opp in self.logs))
        self.allocator.update(contexts[won_mask], items[won_mask], outcomes[won_mask], self.name)

        # Update bidding model with all data
        self.bidder.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, self.name)

        self.truncate_memory()

    def truncate_memory(self):
        if self.memory!='inf' and len(self.logs)>self.memory:
            self.logs = self.logs[-self.memory:]
            self.bidder.b = self.bidder.b[-self.memory:]

    def get_allocation_regret(self):
        ''' How much value am I missing out on due to suboptimal allocation? '''
        return np.sum(list(opp.best_expected_value - opp.true_CTR * opp.value for opp in self.logs[self.record_index:]))

    # def get_estimation_regret(self):
    #     ''' How much am I overpaying due to over-estimation of the value? '''
    #     return np.mean(list(opp.estimated_CTR * opp.value - opp.true_CTR * opp.value for opp in self.logs[self.record_index:]))

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
    
    # def get_gross_utility(self):
    #     return np.sum(list(opp.gross_utility for opp in self.logs[self.record_index:]))

    def get_bid(self):
        return np.array(self.bidder.b[self.record_index:])

    def get_winning_prob(self):
        return np.mean(list(opp.won for opp in self.logs[self.record_index:]))
    
    def get_CTRs(self):
        return np.array(list(opp.true_CTR for opp in self.logs[self.record_index:]))

    def get_optimal_selection_rate(self):
        return np.mean(list(float(opp.optimal_item) for opp in self.logs[self.record_index:]))
    
    def get_bidding_error(self):
        return np.array(list(opp.bidding_error for opp in self.logs[self.record_index:]))

    def get_matrix(self):
        return self.allocator.model.M.numpy(force=True) * np.sqrt(self.feature_dim * self.context_dim)
    
    def get_item_selection(self):
        return self.item_selection.copy()
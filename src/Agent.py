import numpy as np

from BidderAllocation import *
from Impression import ImpressionOpportunity
from Models import sigmoid
from Bidder import TruthfulBidder


class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, num_items, item_values, allocator, bidder, context_dim, update_schedule, memory):
        self.rng = rng
        self.name = name
        self.num_items = num_items
        self.context_dim = context_dim

        # Value distribution
        self.item_values = item_values

        self.logs = []

        self.allocator = allocator
        self.bidder = bidder
        
        self.clock = 0
        self.record_index = 0
        if update_schedule.split()[0]=='doubling':
            self.update_schecule = 'doubling'
            self.exploration_steps = int(update_schedule.split()[1])
        elif update_schedule.split()[0]=='even':
            self.update_schecule = 'even'
            self.update_interval = int(update_schedule.split()[1])
        self.memory = memory

        self.bidding_variance = []
    
    def should_explore(self):
        if self.update_schecule!='doubling':
            return False
        b = 1
        while self.clock >= b:
            b = b<<1
        return self.clock - b>>1 < self.exploration_steps
    
    def should_update(self):
        if self.update_schecule=='doubling':
            b = 1
            while self.clock >= b:
                b = b<<1
            return self.clock == b>>1 and b>=1024
        else:
            return self.clock%self.update_interval==0

    def select_item(self, context):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs = self.allocator.estimate_CTR(context, UCB=True)
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs = self.allocator.estimate_CTR(context, TS=True)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item]

    def bid(self, context):
        # First, pick what item we want to choose
        best_item, estimated_CTR = self.select_item(context)

        # Sample value for this item
        value = self.item_values[best_item]

        if isinstance(self.allocator, OracleAllocator):
            context =context[:self.context_dim]

        if self.should_explore():
            gamma = self.rng.uniform(0.1, 1.5)
            if not isinstance(self.bidder, TruthfulBidder):
                self.bidder.gammas.append(gamma)
            bid, variance = gamma*value, 0.0
        else:
            bid, variance = self.bidder.bid(value, context, estimated_CTR, self.clock)
            # if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            #     mean_CTR = self.allocator.estimate_CTR(context, UCB=False)
            #     estimated_CTR = mean_CTR[best_item]
            # elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            #     estimated_CTR = self.allocator.estimate_CTR(context, TS=False)

        # Log what we know so far
        self.logs.append(ImpressionOpportunity(context=context,
                                               item=best_item,
                                               estimated_CTR=estimated_CTR,
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
                                               gross_utility=0.0))
        
        self.bidding_variance.append(variance)
        self.clock += 1

        return bid, best_item

    def charge(self, price, second_price, outcome):
        self.logs[-1].set_price_outcome(price, second_price, outcome, won=True)
        last_value = self.logs[-1].value * outcome
        self.logs[-1].utility += last_value - price
        self.logs[-1].gross_utility += last_value

    def set_price(self, price):
        self.logs[-1].set_price(price)

    def update(self):
        # update schedule here
        if not self.should_update():
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

        if self.memory!='inf' and len(self.logs)>self.memory:
            self.logs = self.logs[-self.memory:]
            self.bidder.gammas = self.bidder.gammas[-self.memory:]

    def get_allocation_regret(self):
        ''' How much value am I missing out on due to suboptimal allocation? '''
        return np.mean(list(opp.best_expected_value - opp.true_CTR * opp.value for opp in self.logs[self.record_index:]))

    def get_estimation_regret(self):
        ''' How much am I overpaying due to over-estimation of the value? '''
        return np.mean(list(opp.estimated_CTR * opp.value - opp.true_CTR * opp.value for opp in self.logs[self.record_index:]))

    def get_overbid_regret(self):
        ''' How much am I overpaying because I could shade more? '''
        return np.mean(list((opp.price - opp.second_price) * opp.won for opp in self.logs[self.record_index:]))

    def get_underbid_regret(self):
        ''' How much have I lost because I could have shaded less? '''
        # The difference between the winning price and our bid -- for opportunities we lost, and where we could have won without overpaying
        # Important to mention that this assumes a first-price auction! i.e. the price is the winning bid
        return np.mean(list((opp.price - opp.bid) * (not opp.won) * (opp.price < (opp.true_CTR * opp.value)) for opp in self.logs[self.record_index:]))

    def get_CTR_RMSE(self):
        return np.sqrt(np.mean(list((opp.true_CTR - opp.estimated_CTR)**2 for opp in self.logs[self.record_index:])))

    def get_CTR_bias(self):
        return np.mean(list((opp.estimated_CTR / opp.true_CTR) for opp in filter(lambda opp: opp.won, self.logs[self.record_index:])))
    
    def get_uncertainty(self):
        return self.allocator.get_uncertainty()
    
    # def get_bidding_var(self):
    #     var = np.array(self.bidding_variance)
    #     self.bidding_variance = []
    #     return np.sqrt(np.sum(var**2)/var.shape[0])
    
    def move_index(self):
        self.record_index = len(self.logs)

    def get_net_utility(self):
        return np.sum(list(opp.utility for opp in self.logs[self.record_index:]))
    
    def get_gross_utility(self):
        return np.sum(list(opp.gross_utility for opp in self.logs[self.record_index:]))

    def get_gamma(self):
        return np.array(self.bidder.gammas[self.record_index:])

    def get_winning_prob(self):
        return np.mean(list(opp.won for opp in self.logs[self.record_index:]))
    
    def get_CTRs(self):
        return np.array(list(opp.true_CTR for opp in self.logs[self.record_index:]))
import numpy as np
from dataclasses import dataclass

@dataclass
class ImpressionOpportunity:
    __slots__ = ['context', 'item', 'value', 'bid', 'best_expected_value', 'true_CTR', 'estimated_CTR',
                 'optimistic_CTR', 'price', 'second_price', 'winning_bid', 'outcome', 'won', 'utility',
                 'optimal_item', 'bidding_error','auction_no']

    context: np.array
    item: np.uint32
    value: np.float32
    bid: np.float32
    best_expected_value: np.float32
    true_CTR: np.float32
    estimated_CTR: np.float32
    optimistic_CTR: np.float32
    price: np.float32
    second_price: np.float32
    outcome: np.bool
    won: np.bool
    utility: np.float32
    optimal_item: np.bool
    bidding_error: np.float32
    auction_no: np.uint32

    def set_true_CTR(self, best_expected_value, true_CTR):
        self.best_expected_value = best_expected_value  # Best possible CTR (to compute regret from ad allocation)
        self.true_CTR = true_CTR  # True CTR for the chosen ad
        self.optimal_item = self.true_CTR*self.value +1e-9 > self.best_expected_value # true ctr * expected value > true ctr * true item value

    def set_price_outcome(self, price, second_price, outcome, won=True):
        self.price = price
        self.second_price = second_price
        self.outcome = outcome
        self.won = won

    def set_price(self, price):
        self.price = price

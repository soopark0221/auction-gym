import numpy as np
import torch
from tqdm import tqdm

from Impression import ImpressionOpportunity
from Models import *


class Bidder:
    """ Bidder base class"""
    def __init__(self, rng):
        self.rng = rng
        self.truthful = False # Default
        self.item_values = None
        self.b = []
        self.propensity = []

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        pass

    def clear_logs(self, memory):
        pass

    def get_uncertainty(self):
        return np.array([0])
    
    def clear_logs(self, memory):
        if memory=="0":
            self.b = []
            self.propensity = []
        elif memory!='inf' and len(self.b)>int(memory):
            self.b = self.b[-int(memory):]
            self.propensity = self.propensity[-int(memory):]


class TruthfulBidder(Bidder):
    """ A bidder that bids truthfully """
    def __init__(self, rng, noise=0.1, bias = 0.8):
        super(TruthfulBidder, self).__init__(rng)
        self.truthful = True
        self.noise = noise
        self.bias = bias

    def bid(self, value, context, estimated_CTR):
        bid = value * (estimated_CTR * self.bias + self.rng.normal(0,self.noise,1))
        return bid.item()

class OracleBidder(Bidder):
    def __init__(self, rng, optimism_scale=1.0, overbidding_factor=0.0, pessimism_ratio=0.1):
        super().__init__(rng)

    def bid(self, value, estimated_CTR, prob_win, b_grid):
        # Compute the bid as expected value
        expected_value = value * estimated_CTR
        # Grid search over gamma
    
        estimated_utility = prob_win * (expected_value - b_grid)
        bid = b_grid[np.argmax(estimated_utility)]

        self.b.append(bid)
        return bid

class DefaultBidder(Bidder):
    def __init__(self, rng, lr, context_dim, optimism_scale=1.0, noise=0.0, eq_winning_rate=1.0, overbidding_factor=0.0, pessimism_ratio=0.1, overbidding_steps=1e4):
        super().__init__(rng)
        self.lr = lr
        self.context_dim = context_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.winrate_model = NeuralWinRateEstimator(context_dim).to(self.device)
        self.noise = noise
        self.optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, amsgrad=True)

        self.G_w = np.eye(self.context_dim)
        self.G_l = np.eye(self.context_dim)
        self.optimism_scale = optimism_scale
        self.overbidding_factor = overbidding_factor
        self.pessimism_ratio = pessimism_ratio
        self.overbidding_steps = overbidding_steps
        self.clock = 1
    
    def initialize(self, item_values):
        self.item_values = item_values
        X = []
        y = []
        median_value = np.median(self.item_values)
        max_value = np.max(self.item_values)
        for i in range(500):
            context = self.rng.normal(0.0, 1.0, size=self.context_dim)
            b = np.linspace(0.0, max_value , 10).reshape(-1,1)
            y.append(np.clip(b/median_value, 0.0, 1.0))
            X.append(np.concatenate([np.tile(context, (10,1)), b], axis=-1))
        X_init = np.concatenate(X)
        y_init = np.concatenate(y)
        X = torch.Tensor(X_init).to(self.device)
        y = torch.Tensor(y_init).to(self.device)

        epochs = 10000
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        MSE = nn.MSELoss()
        self.winrate_model.train()
        for epoch in tqdm(range(epochs), desc='initializing winrate estimators'):
            optimizer.zero_grad()
            y_pred = self.winrate_model(X)
            loss = MSE(y_pred, y)
            loss.backward()
            optimizer.step()
        self.winrate_model.eval()

    def bid(self, value, context, mean_CTR, uncertainty):
        # Grid search over gamma
        n_values_search = int(value*100)
        b_grid = np.linspace(0.1*value, 1.5*value,n_values_search)
        x = torch.Tensor(np.hstack([np.tile(context, ((n_values_search, 1))), b_grid.reshape(-1,1)])).to(self.device)

        prob_win = self.winrate_model(x).numpy(force=True).ravel()
        won_count = context @ self.G_w @ context / np.sum(context**2)
        optimistic_value = value * (mean_CTR + self.optimism_scale * uncertainty)

        # induce overbidding to collect data
        if self.rng.uniform(0,1) > self.pessimism_ratio:
            estimated_utility = prob_win * (optimistic_value - b_grid)
            bid = b_grid[np.argmax(estimated_utility)]
            bid += self.overbidding_factor * value * np.sqrt(1/(won_count+1e-2))
        else:
            estimated_utility = prob_win * (value * mean_CTR - b_grid)
            bid = b_grid[np.argmax(estimated_utility)]

        bid = np.clip(bid, 0.0, 1.5*value)

        self.b.append(bid)
        self.clock += 1
        return bid, optimistic_value/value

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        if not won_mask.astype(np.uint8).sum():
            print(f'Not enough data collected for {name}')
            return
        N = len(bids)
        won_context = contexts[won_mask]
        won_context = won_context / np.sqrt(np.sum(won_context**2, axis=1, keepdims=True))
        lost_context = contexts[~won_mask]
        lost_context = lost_context / np.sqrt(np.sum(lost_context**2, axis=1, keepdims=True))
        self.G_w =  won_context.T @ won_context
        self.G_l = lost_context.T @ lost_context

        X = np.hstack((contexts.reshape(-1,self.context_dim), bids.reshape(-1, 1)))
        X = torch.Tensor(X).to(self.device)

        y = won_mask.astype(np.float32).reshape(-1,1)
        y = torch.Tensor(y).to(self.device)

        self.winrate_model.train()
        epochs = 500
        for epoch in range(int(epochs)):
            self.optimizer.zero_grad()
            loss = self.winrate_model.loss(X, y)
            loss.backward()
            self.optimizer.step()
        self.winrate_model.eval()
        
    def copy_param(self, ref):
        self.winrate_model.linear1.weight.data = ref.winrate_model.linear1.weight.clone().detach()
        self.winrate_model.linear1.bias.data = ref.winrate_model.linear1.bias.clone().detach()
        self.winrate_model.linear2.weight.data = ref.winrate_model.linear2.weight.clone().detach()
        self.winrate_model.linear2.bias.data = ref.winrate_model.linear2.bias.clone().detach()

class IPSBidder(Bidder):
    def __init__(self, rng, lr, context_dim, entropy_factor, explore_then_commit=None, use_WIS=False, weight_clip=1e4,
                 optimism_scale=1.0, overbidding_factor=0.0, pessimism_ratio=0.1):
        super().__init__(rng)
        self.context_dim = context_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.bidding_policy = StochasticPolicy(context_dim, 'REINFORCE', use_WIS=use_WIS, entropy_factor=entropy_factor, weight_clip=weight_clip).to(self.device)
        self.optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)

        self.explore_then_commit = explore_then_commit
        self.update_interval = 2000

    def initialize(self, item_values):
        self.item_values = item_values

        # initialize bidding policy
        X = []
        max_value = np.max(self.item_values)
        v = np.linspace(0.0, max_value , 10).reshape(-1,1)
        for i in range(500):
            context = self.rng.normal(0.0, 1.0, size=self.context_dim)
            X.append(np.concatenate([np.tile(context.reshape(1,-1), (10,1)), v], axis=1))
        X_init = np.concatenate(X)
        mu_init = np.tile(v*1.0, (500,1))
        X = torch.Tensor(X_init).to(self.device)
        mu = torch.Tensor(mu_init).to(self.device)
        std = mu.clone() * 0.02

        epochs = 10000
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        self.bidding_policy.train()
        MSE = nn.MSELoss()
        for epoch in tqdm(range(epochs), desc='initializing bidding policy'):
            optimizer.zero_grad()
            loss = MSE(self.bidding_policy.mu(X).squeeze(), mu.squeeze())
            loss += MSE(self.bidding_policy.sigma(X).squeeze(), std.squeeze())
            loss.backward()
            optimizer.step()
        self.bidding_policy.eval()

    def bid(self, value, context, estimated_CTR, uncertainty):
        expected_value = value * estimated_CTR
        x = torch.Tensor(np.concatenate([context, np.array(expected_value).reshape(-1)])).to(self.device)
        C = torch.Tensor(context).to(self.device)
        V = torch.Tensor(np.array(estimated_CTR*value)).to(self.device)
        with torch.no_grad():
            bid = self.bidding_policy(x)
            self.propensity.append(self.bidding_policy.normal_pdf(C.reshape(1,-1), V.reshape(1,1), bid).item())
        return np.clip(bid.numpy(force=True), value*0.1, value*1.5).item(), estimated_CTR

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        X = torch.Tensor(contexts[-self.update_interval:]).to(self.device)
        U = torch.Tensor(utilities[-self.update_interval:]).to(self.device)
        V = torch.Tensor(values[-self.update_interval:]*estimated_CTRs[-self.update_interval:]).to(self.device)
        b = torch.Tensor(bids[-self.update_interval:]).to(self.device)
        logging_pp = torch.Tensor(np.array(self.propensity[-self.update_interval:])).to(self.device)

        self.bidding_policy.train()
        N = X.size(0)
        batch_size = N
        epochs = 2000
        for epoch in range(epochs):
            indices = np.arange(N)
            for i in range(int(N/batch_size)):
                ind = indices[i*batch_size:(i+1)*batch_size]
                self.optimizer.zero_grad()
                loss = self.bidding_policy.loss(X[ind], V[ind], b[ind], logging_pp=logging_pp[ind], utility=U[ind])
                loss.backward()
                self.optimizer.step()
        self.bidding_policy.eval()

class DRBidder(Bidder):
    def __init__(self, rng, lr, context_dim, entropy_factor=None, weight_clip=1e4, optimism_scale=1.0, overbidding_factor=0.0, pessimism_ratio=0.1):
        super().__init__(rng)
        self.context_dim = context_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.winrate_model = NeuralWinRateEstimator(context_dim).to(self.device)
        self.winrate_optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)

        self.bidding_policy = StochasticPolicy(context_dim, 'DR', entropy_factor=entropy_factor, weight_clip=weight_clip).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)

        self.update_interval = 2000
    
    def initialize(self, item_values):
        self.item_values = item_values
        max_value = np.max(self.item_values)
        # initialize bidding policy
        X = []
        v = np.linspace(0.0, max_value , 10).reshape(-1,1)
        for i in range(500):
            context = self.rng.normal(0.0, 1.0, size=self.context_dim)
            X.append(np.concatenate([np.tile(context.reshape(1,-1), (10,1)), v], axis=-1))
        X_init = np.concatenate(X)
        mu_init = np.tile(v, (500,1))
        X = torch.Tensor(X_init).to(self.device)
        mu = torch.Tensor(mu_init).to(self.device)
        std = mu.clone() * 0.02

        epochs = 10000
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        self.bidding_policy.train()
        MSE = nn.MSELoss()
        for epoch in tqdm(range(epochs), desc='initializing bidding policy'):
            optimizer.zero_grad()
            loss = MSE(self.bidding_policy.mu(X).squeeze(), mu.squeeze())
            loss += MSE(self.bidding_policy.sigma(X).squeeze(), std.squeeze())
            loss.backward()
            optimizer.step()
        self.bidding_policy.eval()

    def bid(self, value, context, estimated_CTR, uncertainty):
        expected_value = value * estimated_CTR
        x = torch.Tensor(np.concatenate([context, np.array(expected_value).reshape(-1)])).to(self.device)
        C = torch.Tensor(context).to(self.device)
        V = torch.Tensor(np.array(value*estimated_CTR)).to(self.device)
        with torch.no_grad():
            bid = self.bidding_policy(x)
            self.propensity.append(self.bidding_policy.normal_pdf(C.reshape(1,-1), V.reshape(1,1), bid).item())
        return np.clip(bid.numpy(force=True), value*0.1, value*1.5).item(), estimated_CTR

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        # update winrate estimator
        X = np.hstack((contexts.reshape(-1,self.context_dim), bids.reshape(-1, 1)))
        X = torch.Tensor(X).to(self.device)

        y = won_mask.astype(np.float32).reshape(-1,1)
        y = torch.Tensor(y).to(self.device)

        self.winrate_model.train()
        epochs = 2000
        for epoch in range(int(epochs)):
            self.winrate_optimizer.zero_grad()
            loss = self.winrate_model.loss(X, y)
            loss.backward()
            self.winrate_optimizer.step()
        self.winrate_model.eval()

        #update policy
        X = torch.Tensor(contexts[-self.update_interval:]).to(self.device)
        U = torch.Tensor(utilities[-self.update_interval:]).to(self.device)
        V = torch.Tensor(values[-self.update_interval:]*estimated_CTRs[-self.update_interval:]).to(self.device)
        b = torch.Tensor(bids[-self.update_interval:]).to(self.device)
        logging_pp = torch.Tensor(np.array(self.propensity[-self.update_interval:])).to(self.device)

        self.bidding_policy.train()
        self.winrate_model.requires_grad_(False)
        N = X.size(0)
        batch_size = N
        epochs = 2000
        for epoch in range(epochs):
            indices = np.arange(N)
            for i in range(int(N/batch_size)):
                ind = indices[i*batch_size:(i+1)*batch_size]
                self.policy_optimizer.zero_grad()
                loss = self.bidding_policy.loss(X[ind], V[ind], b[ind], logging_pp=logging_pp[ind], utility=U[ind], winrate_model=self.winrate_model)
                loss.backward()
                self.policy_optimizer.step()
        self.winrate_model.requires_grad_(True)

class DMBidder(Bidder):
    def __init__(self, rng, lr, context_dim, optimism_scale=1.0, noise=0.0, eq_winning_rate=1.0, overbidding_factor=0.0, pessimism_ratio=0.1, overbidding_steps=1e4):
        super().__init__(rng)
        self.lr = lr
        self.context_dim = context_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.winrate_model = NeuralWinRateEstimator(context_dim).to(self.device)
        self.noise = noise
        self.optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, amsgrad=True)

        self.optimism_scale = optimism_scale
        self.overbidding_factor = overbidding_factor
        self.pessimism_ratio = pessimism_ratio
        self.update_interval = 2000
    
    def initialize(self, item_values):
        pass

    def bid(self, value, context, mean_CTR, uncertainty):
        # Grid search over gamma
        n_values_search = int(value*100)
        b_grid = np.linspace(0.1*value, 1.5*value,n_values_search)
        x = torch.Tensor(np.hstack([np.tile(context, ((n_values_search, 1))), b_grid.reshape(-1,1)])).to(self.device)

        prob_win = self.winrate_model(x).numpy(force=True).ravel()
        estimated_utility = prob_win * (value * mean_CTR - b_grid)
        bid = b_grid[np.argmax(estimated_utility)]

        bid = np.clip(bid, 0.0, 1.5*value)

        self.b.append(bid)
        return bid, mean_CTR

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        if not won_mask.astype(np.uint8).sum():
            print(f'Not enough data collected for {name}')
            return

        X = np.hstack((contexts.reshape(-1,self.context_dim), bids.reshape(-1, 1)))
        X = torch.Tensor(X).to(self.device)

        y = won_mask.astype(np.float32).reshape(-1,1)
        y = torch.Tensor(y).to(self.device)

        self.winrate_model.train()
        epochs = 2000
        for epoch in range(int(epochs)):
            self.optimizer.zero_grad()
            loss = self.winrate_model.loss(X, y)
            loss.backward()
            self.optimizer.step()
        self.winrate_model.eval()


class RichBidder(Bidder):
    def __init__(self, rng):
        super().__init__(rng)

    def bid(self, value, context, mean_CTR, uncertainty):
        self.b.append(value*2.0)
        return value * 2.0, mean_CTR
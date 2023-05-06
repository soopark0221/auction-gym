import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Impression import ImpressionOpportunity
from Models import *
from swag_misc import SWAG


class Bidder:
    """ Bidder base class"""
    def __init__(self, rng):
        self.rng = rng
        self.truthful = False # Default
        self.item_values = None

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        pass

    def clear_logs(self, memory):
        pass

    def get_uncertainty(self):
        return np.array([0])


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

class DQNBidder(Bidder):
    def __init__(self, rng, lr, gamma_mu, gamma_sigma, context_dim, exploration_method, epsilon=0.1, noise=0.02, prior_var=1.0):
        super().__init__(rng)
        self.lr = lr
        self.gamma_mu = gamma_mu
        self.gamma_sigma = gamma_sigma
        self.context_dim = context_dim
        assert exploration_method in ['Epsilon-greedy', 'Gaussian Noise', 'Bayes by Backprop', 'MC Dropout',
                          'NoisyNet']
        self.method = exploration_method
        self.b = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.method=='Epsilon-greedy':
            self.winrate_model = NeuralWinRateEstimator(context_dim)
            self.eps = epsilon
        elif self.method=='Gaussian Noise':
            self.winrate_model = NeuralWinRateEstimator(context_dim)
            self.noise = noise
        elif self.method=='Bayes by Backprop':
            self.winrate_model = BBBWinRateEstimator(context_dim)
            self.prior_var = prior_var
            
        elif self.method=='MC Dropout':
            self.winrate_model = MCDropoutWinRateEstimator(context_dim)
        elif self.method=='NoisyNet':
            self.winrate_model = NoisyNetWinRateEstimator(context_dim)
        self.winrate_model.to(self.device)
        self.model_initialised = False
        self.count=0
        # self.initialize()
    
    def initialize(self):
        X = []
        y = []
        for i in range(500):
            context = self.rng.normal(0.0, 1.0, size=self.context_dim)
            b = self.rng.uniform(0.0, 1.5, 10).reshape(-1,1)
            y.append((1 + np.exp(-10*(b-1.0)))**(-1))
            X.append(np.concatenate([np.tile(context/np.sqrt(np.sum(context**2)), (10,1)), b], axis=-1))
        self.X_init = np.concatenate(X)
        self.y_init = np.concatenate(y)
        X = torch.Tensor(self.X_init).to(self.device)
        y = torch.Tensor(self.y_init).to(self.device)

        epochs = 100000
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

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        expected_value = value * estimated_CTR
        # Grid search over gamma
        n_values_search = int(value*100)
        b_grid = np.linspace(0.1*value, 1.5*value,n_values_search)
        x = torch.Tensor(np.hstack([np.tile(context, ((n_values_search, 1))), b_grid.reshape(-1,1)])).to(self.device)

        if self.method=='Bayes by Backprop' or self.method=='NoisyNet':
            prob_win = self.winrate_model(x, True).numpy(force=True).ravel()
        elif self.method=='MC Dropout':
            with torch.no_grad():
                prob_win = self.winrate_model(x).numpy(force=True).ravel()
        else:
            prob_win = self.winrate_model(x).numpy(force=True).ravel()

        estimated_utility = prob_win * (expected_value - b_grid)
        bid = b_grid[np.argmax(estimated_utility)]
        
        if self.method=='Epsilon-greedy' and self.rng.random()<self.eps:
            bid = self.rng.uniform(0.1*value,1.5*value)
        elif self.method=='Gaussian Noise':
            bid = np.clip(bid+self.rng.normal(0,self.noise)*value, 0.1*value, 1.5*value)

        self.b.append(bid)
        return bid, 0.0

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        self.count += 1
        # FALLBACK: if you lost every auction you participated in, your model collapsed
        # Revert to not shading for 1 round, to collect data with informational value
        if not won_mask.astype(np.uint8).sum():
            self.model_initialised = False
            print(f'! Fallback for {name}')
            return

        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        utilities = torch.Tensor(utilities).to(self.device)

        # Augment data with samples: if you shade 100%, you will lose
        # If you won now, you would have also won if you bid higher
        X = np.hstack((contexts.reshape(-1,self.context_dim), bids.reshape(-1, 1)))
        # X = np.concatenate([X, self.X_init])
        N = X.shape[0]

        # X_aug_neg = X.copy()
        # X_aug_neg[:, -1] = 0.0
        # X = torch.Tensor(np.vstack((X, X_aug_neg))).to(self.device)

        X = torch.Tensor(X).to(self.device)

        y = won_mask.astype(np.float32).reshape(-1,1)
        # y = np.concatenate([y, self.y_init])
        y = torch.Tensor(y).to(self.device)
        # y = torch.Tensor(np.concatenate((y, np.zeros_like(y)))).to(self.device)

        self.winrate_model.train()
        # epochs = 100
        # optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        # B = min(8192, N)
        # # batch_num = int(N/B)
        # losses = []
        # for epoch in range(int(epochs)):
        #     for i in range(batch_num):
        #         X_mini = X[i:i+B]
        #         y_mini = y[i:i+B]
        #         optimizer.zero_grad()

        #         if self.method=='Bayes by Backprop':
        #             loss = self.winrate_model.loss(X_mini, y_mini, N, 2, self.prior_var)
        #         elif self.method=='NoisyNet':
        #             loss = self.winrate_model.loss(X_mini, y_mini, 2)
        #         else:
        #             loss = self.winrate_model.loss(X_mini, y_mini)
        #         loss.backward()
        #         optimizer.step()
        #     if epoch > 20 and np.abs(losses[-20] - losses[-1]) < 1e-6:
        #         print(f'Stopping at Epoch {epoch}')
        #         break
        epochs = 100
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            if self.method=='Bayes by Backprop':
                loss = self.winrate_model.loss(X, y, N, 2, self.prior_var)
            elif self.method=='NoisyNet':
                loss = self.winrate_model.loss(X, y, 2)
            else:
                loss = self.winrate_model.loss(X, y)
            loss.backward()
            optimizer.step()
        self.winrate_model.eval()
        self.model_initialised = True

    def clear_logs(self, memory):
        if memory=='inf':
            pass
        else:
            self.b = self.b[-memory:]
    
    def get_uncertainty(self):
        if self.method in ['NoisyNet', 'Bayes by Backprop']:
            return self.winrate_model.get_uncertainty()
        else:
            return np.array([0])

class StochasticPolicyBidder(Bidder):
    # work in progress. this will not work.
    def __init__(self, rng, gamma_mu, gamma_sigma, context_dim, loss_type, exploration_method, use_WIS=True, entropy_factor=0.1):
        super().__init__(rng)
        self.gamma_mu = gamma_mu
        self.gamma_sigma = gamma_sigma
        self.context_dim = context_dim + 1
        self.gammas = []
        self.propensities = []
        
        self.MAP_propensity = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert loss_type in ['REINFORCE', 'Actor-Critic', 'PPO-MC', 'PPO-AC', 'DR']
        self.loss_type = loss_type

        if loss_type in ['Actor-Critic', 'PPO-AC', 'DR']:
            self.winrate_model = NeuralWinRateEstimator(context_dim)
            self.winrate_model.to(self.device)
        else:
            self.winrate_model = None


        assert exploration_method in ['Entropy Regularization', 'NoisyNet', 'SWAG', 'MC Dropout']
        self.exploration_method = exploration_method

        if self.exploration_method=='NoisyNet':
            self.bidding_policy = BayesianStochasticPolicy(context_dim, loss_type, use_WIS=use_WIS)
        elif self.exploration_method=='SWAG':
            self.bidding_policy = StochasticPolicy(context_dim, loss_type, use_WIS=use_WIS)
            self.swag_policy = SWAG(self.bidding_policy)
        elif self.exploration_method=='MC Dropout':
            self.bidding_policy = StochasticPolicy(context_dim, loss_type, dropout=0.8, use_WIS=use_WIS)
        else:
            self.bidding_policy = StochasticPolicy(context_dim, loss_type, use_WIS=use_WIS, entropy_factor=entropy_factor)
        self.bidding_policy.to(self.device)
        self.model_initialised = False

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shading factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.gamma_mu, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.gamma_mu - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
            variance = np.array([self.gamma_sigma])
        else:
            # Option 2:
            # Sample from the contextual bandit
            x = torch.Tensor(np.concatenate([context, np.array(estimated_CTR).reshape(1), np.array(value).reshape(1)])).to(self.device)
            with torch.no_grad():
                if self.exploration_method=='NoisyNet':
                    gamma, propensity, variance = self.bidding_policy(x, self.MAP_propensity)
                elif self.exploration_method=='SWAG':
                    self.swag_policy.sample()
                    gamma, propensity, variance = self.swag_policy(x)
                    if self.MAP_propensity:
                        self.swag_policy.sample(add_swag=False)
                        _, propensity = self.bidding_policy.normal_pdf(x, gamma)
                else:
                    gamma, propensity, variance = self.bidding_policy(x)
                gamma = torch.clip(gamma, 0.0, 1.5)

        if self.model_initialised:
            gamma = gamma.detach().item()
        bid *= gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid, variance.reshape(-1)

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        ##############################
        # 1. TRAIN UTILITY ESTIMATOR #
        ##############################
        if self.winrate_model is not None:
            gammas_numpy = np.array(self.gammas)

            # Augment data with samples: if you shade 100%, you will lose
            # If you won now, you would have also won if you bid higher
            X = np.hstack((contexts.reshape(-1,self.context_dim), estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1)))
            N = X.shape[0]

            X_aug_neg = X.copy()
            X_aug_neg[:, -1] = 0.0

            X_aug_pos = X[won_mask].copy()
            X_aug_pos[:, -1] = np.maximum(X_aug_pos[:, -1], 1.0)

            X = torch.Tensor(np.vstack((X, X_aug_neg))).to(self.device)

            y = won_mask.astype(np.uint8).reshape(-1,1)
            y = torch.Tensor(np.concatenate((y, np.zeros_like(y)))).to(self.device)

            # Fit the model
            self.winrate_model.train()
            best_epoch, best_loss = -1, np.inf
            epochs = 10000
            lr = 1e-3
            optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=256, min_lr=1e-7, factor=0.2, verbose=True)
            losses = []
            B = N
            batch_num = int(N/B)

            for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
                epoch_loss = 0

                for i in range(batch_num):
                    X_mini = X[i:i+B]
                    y_mini = y[i:i+B]
                    optimizer.zero_grad()
                    loss = self.winrate_model.loss(X_mini, y_mini)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    scheduler.step(loss)
                losses.append(epoch_loss)

                if (best_loss - losses[-1]) > 1e-6:
                    best_epoch = epoch
                    best_loss = losses[-1]
                elif epoch - best_epoch > 500:
                    print(f'Stopping at Epoch {epoch}')
                    break

            self.winrate_model.eval()

        ##############################
        # 2. TRAIN POLICY #
        ##############################
        utilities = torch.Tensor(utilities).to(self.device)
        gammas = torch.Tensor(self.gammas).to(self.device)

        # Prepare features
        X = torch.Tensor(np.hstack((contexts.reshape(-1,self.context_dim), estimated_CTRs.reshape(-1,1), values.reshape(-1,1)))).to(self.device)
        N = X.size()[0]
        V = torch.Tensor(values).to(self.device)
        P = torch.Tensor(prices).to(self.device)

        if not self.model_initialised:
            self.bidding_policy.initialise_policy(X, gammas)

        # Ensure we don't have propensities that are rounded to zero
        propensities = torch.clip(torch.Tensor(self.propensities).to(self.device), min=1e-6)

        # Fit the model
        self.bidding_policy.train()
        if self.winrate_model is not None:
            self.winrate_model.requires_grad_(False)
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, threshold=5e-3, verbose=True)

        losses = []
        B = N
        batch_num = int(N/B)
        best_epoch, best_loss = -1, np.inf

        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            epoch_loss = 0

            for i in range(batch_num):
                X_mini = X[i:i+B]
                V_mini = V[i:i+B]
                P_mini = P[i:i+B]
                gamma_mini = gammas[i:i+B]
                pp_mimi = propensities[i:i+B]
                util_mini = utilities[i:i+B]
                optimizer.zero_grad()
                if self.winrate_model is None:
                    loss = self.bidding_policy.loss(
                        X_mini, V_mini, P_mini, gamma_mini, pp_mimi, util_mini)
                else:
                    loss = self.bidding_policy.loss(
                        X_mini, V_mini, P_mini, gamma_mini, pp_mimi, util_mini, self.winrate_model)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                scheduler.step(loss)
            losses.append(epoch_loss)
            
            if self.exploration_method=='SWAG':
                self.swag_policy.collect_model(self.bidding_policy)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 500:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)
        if np.isnan(losses).any():
            print('NAN DETECTED! in losses')
            print(list(losses))
            print(np.isnan(X.detach().numpy()).any(), X)
            print(np.isnan(gammas.detach().numpy()).any(), gammas)
            print(np.isnan(propensities.detach().numpy()).any(), propensities)
            print(np.isnan(utilities.detach().numpy()).any(), utilities)
            exit(1)

        self.bidding_policy.eval()
        if self.winrate_model is not None:
            self.winrate_model.requires_grad_(True)
        self.model_initialised = True
        self.bidding_policy.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]
    
    def get_uncertainty(self):
        if self.exploration_method=='NoisyNet':
            return self.bidding_policy.get_uncertainty()
        else:
            return np.array([0])

class DDPGBidder(Bidder):

    def __init__(self, rng, lr, gamma_mu, gamma_sigma, context_dim, exploration_method, noise=0.02, prior_var=1.0):
        super().__init__(rng)
        self.lr = lr
        self.gamma_mu = gamma_mu
        self.gamma_sigma = gamma_sigma
        self.context_dim = context_dim
        self.gammas = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.winrate_model = NeuralWinRateEstimator(context_dim).to(self.device)

        assert exploration_method in ['Gaussian Noise', 'NoisyNet', 'Bayes by Backprop', 'SWAG']
        self.exploration_method = exploration_method
        if self.exploration_method=='NoisyNet':
            self.bidding_policy = BayesianDeterministicPolicy(context_dim)
        elif self.exploration_method=='Bayes by Backprop':
            self.bidding_policy = BayesianDeterministicPolicy(context_dim, prior_var)
        elif self.exploration_method=='SWAG':
            self.bidding_policy = DeterministicPolicy(context_dim)
            self.SWAG = SWAG(self.bidding_policy)
        else:
            self.bidding_policy = DeterministicPolicy(context_dim)
            self.noise = noise
        self.bidding_policy.to(self.device)
        self.model_initialised = False

    def bid(self, value, context, estimated_CTR, clock):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shading factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.gamma_mu, self.gamma_sigma)
        else:
            # Option 2:
            # Sample from the contextual bandit
            x = torch.Tensor(np.concatenate([context, np.array(estimated_CTR*value).reshape(1)])).to(self.device)
            with torch.no_grad():
                if self.exploration_method=='SWAG':
                    self.SWAG.sample()
                gamma = self.bidding_policy(x)
        
        if self.model_initialised:
            gamma = gamma.detach().item()
        if self.exploration_method=='Gaussian noise':
            gamma += self.rng.normal(0,self.noise)
        gamma = np.clip(gamma, 0.1, 1.5)
        bid *= gamma
        self.gammas.append(gamma)
        return bid, 0.0

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        ##############################
        # 1. TRAIN UTILITY ESTIMATOR #
        ##############################
        gammas_numpy = np.array(self.gammas)
        X = np.hstack((contexts.reshape(-1, self.context_dim), (estimated_CTRs*values*gammas_numpy).reshape(-1, 1)))
        N = X.shape[0]

        X_aug_neg = X.copy()
        X_aug_neg[:, -1] = 0.0

        X = torch.Tensor(np.vstack((X, X_aug_neg))).to(self.device)

        y = won_mask.astype(np.uint8).reshape(-1,1)
        y = torch.Tensor(np.concatenate((y, np.zeros_like(y)))).to(self.device)

        # Fit the model
        self.winrate_model.train()
        epochs = 10000
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=256, min_lr=1e-7, factor=0.2, verbose=True)
        losses = []
        best_epoch, best_loss = -1, np.inf
        B = min(8192, N)
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            epoch_loss = 0

            for i in range(batch_num):
                X_mini = X[i:i+B]
                y_mini = y[i:i+B]
                optimizer.zero_grad()
                loss = self.winrate_model.loss(X_mini, y_mini)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                scheduler.step(loss)
            losses.append(epoch_loss)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 500:
                print(f'Stopping at Epoch {epoch}')
                break

        self.winrate_model.eval()

        ##############################
        # 2. TRAIN POLICY #
        ##############################
        gammas = torch.Tensor(self.gammas).to(self.device)
        X = torch.Tensor(contexts.reshape(-1,self.context_dim)).to(self.device)
        R = torch.Tensor((outcomes*values).reshape(-1,1)).to(self.device)
        V = torch.Tensor((estimated_CTRs*values).reshape(-1,1)).to(self.device)
        N = X.size()[0]
        
        if not self.model_initialised:
            self.bidding_policy.initialise_policy(torch.cat([X, V], dim=1), gammas)

        # Fit the model
        self.bidding_policy.train()
        epochs = 10000
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, threshold=5e-3, verbose=True)

        losses = []
        best_epoch, best_loss = -1, np.inf
        B = min(8192, N)
        batch_num = int(N/B)

        self.winrate_model.requires_grad_(False)

        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            epoch_loss = 0

            for i in range(batch_num):
                X_mini = X[i:i+B]
                # R_mini = R[i:i+B]
                V_mini = V[i:i+B]
                optimizer.zero_grad()

                if self.exploration_method=='Bayes by Backprop':
                    loss = self.bidding_policy.loss(self.winrate_model, X_mini, V_mini, N)
                else:
                    loss = self.bidding_policy.loss(self.winrate_model, X_mini, V_mini)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                scheduler.step(loss)
            losses.append(epoch_loss)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 500:
                print(f'Stopping at Epoch {epoch}')
                break

        
        losses = np.array(losses)
        if np.isnan(losses).any():
            print('NAN DETECTED! in losses')
            print(losses)
            exit(1)

        self.winrate_model.requires_grad_(True)
        self.bidding_policy.eval()
        self.model_initialised = True
        self.bidding_policy.model_initialised = True
    
    def get_uncertainty(self):
        if self.exploration_method in ['NoisyNet', 'Bayes by Backprop']:
            return self.bidding_policy.get_uncertainty()
        else:
            return np.array([0])

class OracleBidder(Bidder):
    def __init__(self, rng):
        super().__init__(rng)
        self.b = []

    def bid(self, value, estimated_CTR, prob_win, b_grid):
        # Compute the bid as expected value
        expected_value = value * estimated_CTR
        # Grid search over gamma
    
        estimated_utility = prob_win * (expected_value - b_grid)
        bid = b_grid[np.argmax(estimated_utility)]

        self.b.append(bid)
        return bid

    def clear_logs(self, memory):
        if memory=='inf':
            pass
        else:
            self.b = self.b[-memory:]

class DefaultBidder(Bidder):
    def __init__(self, rng, lr, context_dim, optimism_scale=5.0, noise=0.0, eq_winning_rate=2.0, overbidding_factor=5.0, overbidding_steps=1e4):
        super().__init__(rng)
        self.lr = lr
        self.context_dim = context_dim
        self.b = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.winrate_model = NeuralWinRateEstimator(context_dim).to(self.device)
        self.noise = noise
        self.optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, amsgrad=True)

        self.G_w = np.eye(self.context_dim)
        self.G_l = np.eye(self.context_dim)
        self.optimism_scale = optimism_scale
        self.eq_winning_rate = eq_winning_rate
        self.overbidding_factor = overbidding_factor
        self.overbidding_steps = overbidding_steps
    
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

        won_count = context @ self.G_w @ context
        lost_count = context @ self.G_l @ context
        total_count = (won_count + lost_count) / np.sum(context**2)

        # optimism = self.optimism_scale * np.log((self.eq_winning_rate*lost_count+1e-6)/(won_count+1e-6))**3 * np.exp(-1e-3 * total_count)
        optimism = self.optimism_scale * np.exp(-total_count/self.overbidding_steps)
        optimistic_value = value * (mean_CTR + optimism * uncertainty)

        # induce overbidding to collect data
        if self.rng.uniform(0,1) < 0.9:
            estimated_utility = prob_win * (optimistic_value - b_grid)
            bid = b_grid[np.argmax(estimated_utility)]
            bid += self.overbidding_factor * value * (0.1 * np.log((self.eq_winning_rate*lost_count+1e-6)/(won_count+1e-6))**2 + np.sqrt(1/(won_count+1e-2))) * np.exp(-total_count/self.overbidding_steps)
        else:
            estimated_utility = prob_win * (value * mean_CTR - b_grid)
            bid = b_grid[np.argmax(estimated_utility)]

        bid = np.clip(bid+self.rng.normal(0,self.noise)*value, 0.1*value, 1.5*value)

        self.b.append(bid)
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

    def clear_logs(self, memory):
        if memory=='inf':
            pass
        else:
            self.b = self.b[-memory:]
        
    def copy_param(self, ref):
        self.winrate_model.linear1.weight.data = ref.winrate_model.linear1.weight.clone().detach()
        self.winrate_model.linear1.bias.data = ref.winrate_model.linear1.bias.clone().detach()
        self.winrate_model.linear2.weight.data = ref.winrate_model.linear2.weight.clone().detach()
        self.winrate_model.linear2.bias.data = ref.winrate_model.linear2.bias.clone().detach()

class IPSBidder(Bidder):
    def __init__(self, rng, lr, context_dim, entropy_factor, use_WIS=False, weight_clip=1e2):
        super().__init__(rng)
        self.context_dim = context_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.bidding_policy = StochasticPolicy(context_dim, 'REINFORCE', use_WIS=use_WIS, entropy_factor=entropy_factor, weight_clip=weight_clip).to(self.device)
        self.optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        self.b = []
        self.propensity = []

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
        mu_init = np.tile(v, (500,1))
        X = torch.Tensor(X_init).to(self.device)
        mu = torch.Tensor(mu_init).to(self.device)
        std = torch.ones_like(mu).to(self.device) * 0.1 * max_value

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

    def bid(self, value, context, estimated_CTR):
        expected_value = value * estimated_CTR
        x = torch.Tensor(np.concatenate([context, np.array(expected_value).reshape(-1)])).to(self.device)
        C = torch.Tensor(context).to(self.device)
        V = torch.Tensor(np.array(value)).to(self.device)
        with torch.no_grad():
            bid = self.bidding_policy(x)
            self.propensity.append(self.bidding_policy.normal_pdf(C.reshape(1,-1), V.reshape(1,1), bid).item())
        return np.clip(bid.numpy(force=True), value*0.1, value*1.5).item()

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        X = torch.Tensor(contexts).to(self.device)
        U = torch.Tensor(utilities).to(self.device)
        V = torch.Tensor(values*estimated_CTRs).to(self.device)
        b = torch.Tensor(bids).to(self.device)
        logging_pp = torch.Tensor(np.array(self.propensity)).to(self.device)

        self.bidding_policy.train()
        N = X.size(0)
        batch_size = min(N, 512)
        epochs = 10
        for epoch in range(epochs):
            shuffled_ind = self.rng.choice(N, size=N, replace=False)
            for i in range(int(N/batch_size)):
                ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
                self.optimizer.zero_grad()
                loss = self.bidding_policy.loss(X[ind], V[ind], b[ind], logging_pp=logging_pp[ind], utility=U[ind])
                loss.backward()
                self.optimizer.step()
        self.bidding_policy.eval()

    def clear_logs(self, memory):
        if memory=='inf':
            pass
        else:
            self.b = self.b[-memory:]

class DRBidder(Bidder):
    def __init__(self, rng, lr, context_dim, entropy_factor=None, weight_clip=None):
        super().__init__(rng)
        self.context_dim = context_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.winrate_model = NeuralWinRateEstimator(context_dim).to(self.device)
        self.winrate_optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)

        self.bidding_policy = StochasticPolicy(context_dim, 'DR', entropy_factor=entropy_factor, weight_clip=weight_clip).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        self.b = []
        self.propensity = []
    
    def initialize(self, item_values):
        self.item_values = item_values

        # initialize winrate estimator
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
        std = torch.ones_like(mu).to(self.device) * 0.1 * max_value

        epochs = 10000
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        self.bidding_policy.train()
        MSE = nn.MSELoss()
        for epoch in tqdm(range(epochs), desc='initializing bidding policy'):
            optimizer.zero_grad()
            loss = MSE(self.bidding_policy.mu(X).squeeze(), mu.squeeze())
            loss += MSE(self.bidding_policy.std(X).squeeze(), std.squeeze())
            loss.backward()
            optimizer.step()
        self.bidding_policy.eval()

    def bid(self, value, context, estimated_CTR):
        expected_value = value * estimated_CTR
        x = torch.Tensor(np.concatenate([context, np.array(expected_value).reshape(-1)])).to(self.device)
        C = torch.Tensor(context).to(self.device)
        V = torch.Tensor(np.array(value)).to(self.device)
        with torch.no_grad():
            bid = self.bidding_policy(x)
            self.propensity.append(self.bidding_policy.normal_pdf(C.reshape(1,-1), V.reshape(1,1), bid).item())
        return np.clip(bid.numpy(force=True), value*0.1, value*1.5).item()

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, utilities, name):
        # update winrate estimator
        X = np.hstack((contexts.reshape(-1,self.context_dim), bids.reshape(-1, 1)))
        X = torch.Tensor(X).to(self.device)

        y = won_mask.astype(np.float32).reshape(-1,1)
        y = torch.Tensor(y).to(self.device)

        self.winrate_model.train()
        epochs = 100
        for epoch in range(int(epochs)):
            self.winrate_optimizer.zero_grad()
            loss = self.winrate_model.loss(X, y)
            loss.backward()
            self.winrate_optimizer.step()
        self.winrate_model.eval()

        #update policy
        X = torch.Tensor(contexts).to(self.device)
        U = torch.Tensor(utilities).to(self.device)
        V = torch.Tensor(values*estimated_CTRs).to(self.device)
        b = torch.Tensor(bids).to(self.device)
        logging_pp = torch.Tensor(np.array(self.propensity)).to(self.device)

        self.bidding_policy.train()
        self.winrate_model.requires_grad_(False)
        N = X.size(0)
        batch_size = min(N, 512)
        epochs = 10
        for epoch in range(epochs):
            shuffled_ind = self.rng.choice(N, size=N, replace=False)
            for i in range(int(N/batch_size)):
                ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
                self.policy_optimizer.zero_grad()
                loss = self.bidding_policy.loss(X[ind], V[ind], b[ind], logging_pp=logging_pp[ind], utility=U[ind], winrate_model=self.winrate_model)
                loss.backward()
                self.policy_optimizer.step()
        self.winrate_model.requires_grad_(True)

    def clear_logs(self, memory):
        if memory=='inf':
            pass
        else:
            self.b = self.b[-memory:]

class RichBidder(Bidder):
    def __init__(self, rng):
        super().__init__(rng)
        self.b = []

    def bid(self, value, context, mean_CTR, uncertainty):
        self.b.append(value*2.0)
        return value * 2.0, mean_CTR
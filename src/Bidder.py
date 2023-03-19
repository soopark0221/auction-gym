import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Impression import ImpressionOpportunity
from Models import *


class Bidder:
    """ Bidder base class"""
    def __init__(self, rng):
        self.rng = rng
        self.truthful = False # Default

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        pass

    def clear_logs(self, memory):
        pass


class TruthfulBidder(Bidder):
    """ A bidder that bids truthfully """
    def __init__(self, rng):
        super(TruthfulBidder, self).__init__(rng)
        self.truthful = True

    def bid(self, value, context, estimated_CTR):
        return value * estimated_CTR


class EmpiricalShadedBidder(Bidder):
    """ A bidder that learns a single bidding factor gamma from past data """

    def __init__(self, rng, gamma_sigma, init_gamma=1.0):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.gammas = []
        super(EmpiricalShadedBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        # Sample the shade factor gamma
        gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
        if gamma < 0.0:
            gamma = 0.0
        if gamma > 1.0:
            gamma = 1.0
        bid *= gamma
        self.gammas.append(gamma)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        # Extract shading factors to numpy
        gammas = np.array(self.gammas)

        if plot:
            _,_=plt.subplots(figsize=figsize)
            plt.title('Raw observations',fontsize=fontsize+2)
            plt.scatter(gammas,utilities, alpha=.25)
            plt.xlabel(r'Shading factor ($\gamma$)',fontsize=fontsize)
            plt.ylabel('Net Utility',fontsize=fontsize)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)
            # plt.show()

        # We want to be able to estimate utility for any other continuous value, but this is a hassle in continuous space.
        # Instead -- we'll bucketise and look at the empirical utility distribution within every bucket
        min_gamma, max_gamma = np.min(gammas), np.max(gammas)
        grid_delta = .005
        num_buckets = int((max_gamma-min_gamma) // grid_delta) + 1
        buckets = np.linspace(min_gamma, max_gamma, num_buckets)
        x = []
        estimated_y_mean = []
        estimated_y_stderr = []
        bucket_lo = buckets[0]
        for idx, bucket_hi in enumerate(buckets[1:]):
            # Mean of the bucket
            x.append((bucket_hi-bucket_lo)/2.0 + bucket_lo)
            # Only look at samples within this range
            mask = np.logical_and(gammas < bucket_hi, bucket_lo <= gammas)
            # If we can draw meaningful inferences
            num_samples = len(utilities[mask])
            if num_samples > 1:
                # Extrapolate mean utility from these samples
                bucket_utility = utilities[mask].mean()
                estimated_y_mean.append(bucket_utility)
                # Compute standard error on utility estimate
                estimated_y_stderr.append(np.std(utilities[mask]) / np.sqrt(num_samples))
            else:
                estimated_y_mean.append(np.nan)
                estimated_y_stderr.append(np.nan)
            # Move sliding window for bucket
            bucket_lo = bucket_hi
        # To NumPy format
        x = np.asarray(x)
        estimated_y_mean = np.asarray(estimated_y_mean)
        estimated_y_stderr = np.asarray(estimated_y_stderr)

        # This is relatively high because we underestimate total variance
        # (1) Variance from click ~ Bernoulli(p)
        # (2) Variance from uncertainty about winning the auction
        critical_value = 1.96
        U_lower_bound = estimated_y_mean - critical_value * estimated_y_stderr

        # Move the mean of the policy towards the empirically best value
        # Search the array in reverse so we take the highest value in case of ties
        best_idx = len(x) - np.nanargmax(U_lower_bound[::-1]) - 1
        best_gamma = x[best_idx]
        if best_gamma < 0:
            best_gamma = 0
        if best_gamma > 1.0:
            best_gamma = 1.0
        self.prev_gamma = best_gamma

        if plot:
            fig, axes = plt.subplots(figsize=figsize)
            plt.suptitle(name, fontsize=fontsize+2)
            plt.title(f'Iteration: {iteration}', fontsize=fontsize)
            plt.plot(x, estimated_y_mean, label='Estimate', ls='--', color='red')
            plt.fill_between(x,
                             estimated_y_mean - critical_value * estimated_y_stderr,
                             estimated_y_mean + critical_value * estimated_y_stderr,
                             alpha=.25,
                             color='red',
                             label='C.I.')
            plt.axvline(best_gamma, ls='--', color='gray', label='Best')
            plt.axhline(0, ls='-.', color='gray')
            plt.xlabel(r'Multiplicative Bid Shading Factor ($\gamma$)', fontsize=fontsize)
            plt.ylabel('Estimated Net Utility', fontsize=fontsize)
            plt.ylim(-1.0, 2.0)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()
            #plt.show()

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
        else:
            self.gammas = self.gammas[-memory:]


class ValueLearningBidder(Bidder):
    """ A bidder that estimates the optimal bid shading distribution via value learning """

    def __init__(self, rng, gamma_sigma, init_gamma, inference, context_dim, exploration_noise):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.exploration_noise = exploration_noise
        self.context_dim = context_dim + 1
        assert inference in ['search', 'policy']
        self.inference = inference
        self.gammas = []
        self.propensities = []
        self.winrate_model = PyTorchWinRateEstimator(context_dim)
        self.bidding_policy = BidShadingPolicy(context_dim) if inference == 'policy' else None
        self.model_initialised = False
        super(ValueLearningBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shadin factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.prev_gamma - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
        elif self.inference == 'search':
            # Option 2:
            # Predict P(win|gamma,value,P(click))
            # Use it to predict utility, maximise utility
            n_values_search = 128
            gamma_grid = np.linspace(0.1, 1 ,n_values_search)
            x = torch.Tensor(np.hstack((np.tile(context, ((n_values_search, 1))), np.tile(estimated_CTR, (n_values_search, 1)),
                                        np.tile(value, (n_values_search, 1)), gamma_grid.reshape(-1,1))))

            prob_win = self.winrate_model(x).detach().numpy().ravel()

            # U = W (V - P)
            expected_value = bid
            shaded_bids = expected_value * gamma_grid
            estimated_utility = prob_win * (expected_value - shaded_bids)
            gamma = gamma_grid[np.argmax(estimated_utility)]
            gamma += self.exploration_noise * self.rng.normal()
            propensity = 1.0

        elif self.inference == 'policy':
            # Option 3: sample from the learnt policy instead of brute force searching
            x = torch.Tensor(np.concatenate([context, np.array(estimated_CTR).reshape(1,), np.array(value).reshape(1,)]))
            with torch.no_grad():
                gamma, propensity = self.bidding_policy(x)
                gamma = gamma.detach().item()

        bid *= gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # FALLBACK: if you lost every auction you participated in, your model collapsed
        # Revert to not shading for 1 round, to collect data with informational value
        if not won_mask.astype(np.uint8).sum():
            self.model_initialised = False
            print(f'! Fallback for {name}')
            return

        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        utilities = torch.Tensor(utilities)

        # Augment data with samples: if you shade 100%, you will lose
        # If you won now, you would have also won if you bid higher
        X = np.hstack((contexts.reshape(-1,self.context_dim), estimated_CTRs.reshape(-1,1), values.reshape(-1,1), np.array(self.gammas).reshape(-1, 1)))

        X_aug_neg = X.copy()
        X_aug_neg[:, -1] = 0.0

        X_aug_pos = X[won_mask].copy()
        X_aug_pos[:, -1] = np.maximum(X_aug_pos[:, -1], 1.0)

        X = torch.Tensor(np.vstack((X, X_aug_neg)))

        y = won_mask.astype(np.uint8).reshape(-1,1)
        y = torch.Tensor(np.concatenate((y, np.zeros_like(y))))

        # Fit the model
        self.winrate_model.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-7, factor=0.1, verbose=True)
        criterion = torch.nn.BCELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        batch_size = 128
        batch_num = int(8192/128)
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            epoch_loss = 0
            for i in range(batch_num):
                X_mini = X[i:i+batch_size]
                y_mini = y[i:i+batch_size]
                optimizer.zero_grad()
                pred_y = self.winrate_model(X_mini)
                loss = criterion(pred_y, y_mini)/batch_num
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                scheduler.step(loss)
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 100:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)

        self.winrate_model.eval()
        # fig, ax = plt.subplots()
        # plt.title(f'{name}')
        # plt.plot(losses, label=r'P(win|$gamma$,x)')
        # plt.ylabel('Loss')
        # plt.legend()
        # fig.set_tight_layout(True)
        # plt.show()

        # Predict Utility -- \hat{u}
        # orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), np.array(self.gammas).reshape(-1, 1))))
        # W = self.winrate_model(orig_features).squeeze().detach().numpy()
        # print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

        if self.inference == 'policy':
            # Learn a policy to maximise E[U | bid] where bid ~ policy
            X = torch.Tensor(np.hstack((contexts.reshape(-1, self.context_dim), estimated_CTRs.reshape(-1,1), values.reshape(-1,1))))

            self.bidding_policy.train()
            epochs = 10000
            lr = 1e-3
            optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-7, factor=0.1, verbose=True)
            losses = []
            best_epoch, best_loss = -1, np.inf
            batch_size = 128
            batch_num = int(8192/128)
            for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
                epoch_loss = 0
                for i in range(batch_num):
                    X_mini = X[i:i+batch_size]
                    y_mini = y[i:i+batch_size]
                    optimizer.zero_grad()

                    sampled_gamma, propensities = self.bidding_policy(X_mini)
                    X_with_gamma = torch.hstack((X_mini, sampled_gamma))

                    prob_win = self.winrate_model(X_with_gamma).squeeze()
                    values = X_with_gamma[:, self.context_dim].squeeze() * X_with_gamma[:, self.context_dim+1].squeeze()
                    prices = values * sampled_gamma.squeeze()

                    estimated_utility = -(prob_win * (values - prices)).mean()/batch_num
                    estimated_utility.backward()
                    optimizer.step()
                    scheduler.step(estimated_utility)
                    epoch_loss += estimated_utility.item()
                losses.append(epoch_loss)
                if (best_loss - losses[-1]) > 1e-6:
                    best_epoch = epoch
                    best_loss = losses[-1]
                elif epoch - best_epoch > 100:
                    print(f'Stopping at Epoch {epoch}')
                    break

            losses = np.array(losses)
            self.bidding_policy.eval()
            # fig, ax = plt.subplots()
            # plt.title(f'{name}')
            # plt.plot(losses, label=r'$\pi(\gamma)$')
            # plt.ylabel('- Estimated Expected Utility')
            # plt.legend()
            # fig.set_tight_layout(True)
            # #plt.show()

        self.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]


class PolicyLearningBidder(Bidder):
    """ A bidder that estimates the optimal bid shading distribution via policy learning """

    def __init__(self, rng, gamma_mu, gamma_sigma, loss, context_dim):
        self.gamma_mu = gamma_mu
        self.gamma_sigma = gamma_sigma
        self.context_dim = context_dim + 1
        self.gammas = []
        self.propensities = []
        self.model = BidShadingContextualBandit(loss, context_dim)
        self.model_initialised = False
        super(PolicyLearningBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shading factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.prev_gamma - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
        else:
            # Option 2:
            # Sample from the contextual bandit
            x = torch.Tensor([estimated_CTR, value])
            gamma, propensity = self.model(x)
            gamma = torch.clip(gamma, 0.0, 1.0)

        bid *= gamma.detach().item() if self.model_initialised else gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        utilities = torch.Tensor(utilities)

        # Extract shading factors to torch
        gammas = torch.Tensor(self.gammas)

        # Prepare features
        X = torch.Tensor(np.hstack((contexts.shape(-1,self.context_dim), estimated_CTRs.reshape(-1,1), values.reshape(-1,1))))

        if not self.model_initialised:
            # Initialize the model so that it is 'close' to the behavior policy(initial random policy)
            self.model.initialise_policy(X, gammas)

        # Ensure we don't have propensities that are rounded to zero
        propensities = torch.clip(torch.Tensor(self.propensities), min=1e-15)

        # Fit the model
        self.model.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, verbose=True)

        losses = []
        best_epoch, best_loss = -1, np.inf
        batch_size = 128
        batch_num = int(8192/128)

        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            epoch_loss = 0
            for i in range(batch_num):
                X_mini = X[i:i+batch_num,...]
                gammas_mini = gammas[i:i+batch_num]
                optimizer.zero_grad()
                loss = self.model.loss(X_mini, gammas_mini, propensities, utilities, importance_weight_clipping_eps=10.0)/batch_num
                loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
                optimizer.step()  # Updates weights and biases with the optimizer (SGD)
                epoch_loss += loss.item()
                scheduler.step(loss)
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 100:
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

        self.model.eval()
        expected_utility = -self.model.loss(X, gammas, propensities, utilities, KL_weight=0.0).detach().numpy()
        print('Expected utility:', expected_utility)

        pred_gammas, _ = self.model(X)
        pred_gammas = pred_gammas.detach().numpy()
        print(name, 'Number of samples: ', X.shape)
        print(name, 'Predicted Gammas: ', pred_gammas.min(), pred_gammas.max(), pred_gammas.mean())

        self.model_initialised = True
        self.model.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]


class DoublyRobustBidder(Bidder):
    """ A bidder that estimates the optimal bid shading distribution with a Doubly Robust Estimator """

    def __init__(self, rng, gamma_sigma, init_gamma=1.0):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.gammas = []
        self.propensities = []
        self.winrate_model = PyTorchWinRateEstimator()
        self.bidding_policy = BidShadingContextualBandit(loss='Doubly Robust', winrate_model=self.winrate_model)
        self.model_initialised = False
        super(DoublyRobustBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shading factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.prev_gamma - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
        else:
            # Option 2:
            # Sample from the contextual bandit
            x = torch.Tensor([estimated_CTR, value])
            with torch.no_grad():
                gamma, propensity = self.bidding_policy(x)
                gamma = torch.clip(gamma, 0.0, 1.0)

        bid *= gamma.detach().item() if self.model_initialised else gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        # utilities = torch.Tensor(utilities)

        ##############################
        # 1. TRAIN UTILITY ESTIMATOR #
        ##############################
        gammas_numpy = np.array([g.detach().item() if self.model_initialised else g for g in self.gammas])
        if self.model_initialised:
            # Predict Utility -- \hat{u}
            orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1))))
            W = self.winrate_model(orig_features).squeeze().detach().numpy()
            print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

            V = estimated_CTRs * values
            P = estimated_CTRs * values * gammas_numpy
            estimated_utilities = W * (V - P)

            errors = estimated_utilities - utilities
            print('Estimated Utility\t Mean Error:\t\t\t', errors.mean())
            print('Estimated Utility\t Mean Absolute Error:\t', np.abs(errors).mean())

        # Augment data with samples: if you shade 100%, you will lose
        # If you won now, you would have also won if you bid higher
        X = np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1)))

        X_aug_neg = X.copy()
        X_aug_neg[:, -1] = 0.0

        X_aug_pos = X[won_mask].copy()
        X_aug_pos[:, -1] = np.maximum(X_aug_pos[:, -1], 1.0)

        X = torch.Tensor(np.vstack((X, X_aug_neg)))

        y = won_mask.astype(np.uint8).reshape(-1,1)
        y = torch.Tensor(np.concatenate((y, np.zeros_like(y))))

        # Fit the model
        self.winrate_model.train()
        epochs = 8192 * 4
        lr = 3e-3
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=256, min_lr=1e-7, factor=0.2, verbose=True)
        criterion = torch.nn.BCELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()
            pred_y = self.winrate_model(X)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 1024:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)

        self.winrate_model.eval()

        # Predict Utility -- \hat{u}
        orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1))))
        W = self.winrate_model(orig_features).squeeze().detach().numpy()
        print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

        V = estimated_CTRs * values
        P = estimated_CTRs * values * gammas_numpy
        estimated_utilities = W * (V - P)

        errors = estimated_utilities - utilities
        print('Estimated Utility\t Mean Error:\t\t\t', errors.mean())
        print('Estimated Utility\t Mean Absolute Error:\t', np.abs(errors).mean())

        ##############################
        # 2. TRAIN DOUBLY ROBUST POLICY #
        ##############################
        utilities = torch.Tensor(utilities)
        estimated_utilities = torch.Tensor(estimated_utilities)
        gammas = torch.Tensor(self.gammas)

        # Prepare features
        X = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1))))

        if not self.model_initialised:
            self.bidding_policy.initialise_policy(X, gammas)

        # Ensure we don't have propensities that are rounded to zero
        propensities = torch.clip(torch.Tensor(self.propensities), min=1e-15)

        # Fit the model
        self.bidding_policy.train()
        epochs = 8192 * 4
        lr = 7e-3
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, threshold=5e-3, verbose=True)

        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()  # Setting our stored gradients equal to zero
            loss = self.bidding_policy.loss(X, gammas, propensities, utilities, utility_estimates=estimated_utilities, winrate_model=self.winrate_model, importance_weight_clipping_eps=50.0)
            loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
            optimizer.step()  # Updates weights and biases with the optimizer (SGD)
            losses.append(loss.item())
            scheduler.step(loss)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 512:
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

        pred_gammas, _ = self.bidding_policy(X)
        pred_gammas = pred_gammas.detach().numpy()
        print(name, 'Number of samples: ', X.shape)
        print(name, 'Predicted Gammas: ', pred_gammas.min(), pred_gammas.max(), pred_gammas.mean())

        self.model_initialised = True
        self.bidding_policy.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]

class ValueSamplingBidder(Bidder):
    """ A bidder that estimates the utility with its uncertainty via Bayes by Backprop """

    def __init__(self, rng, gamma_mu, gamma_sigma, context_dim, method, epsilon=0.1, noise=0.02, prior_var=1.0):
        self.gamma_mu = gamma_mu
        self.gamma_sigma = gamma_sigma
        self.context_dim = context_dim + 1
        assert method in ['Epsilon-greedy', 'Gaussian Noise', 'Bayes by Backprop', 'MC Dropout',
                          'NoisyNet']
        self.method = method
        self.gammas = []
        if method=='Epsilon-greedy':
            self.winrate_model = NeuralWinRateEstimator(context_dim)
            self.eps = epsilon
        elif method=='Gaussian Noise':
            self.winrate_model = NeuralWinRateEstimator(context_dim)
            self.noise = noise
        elif method=='Bayes by Backprop':
            self.winrate_model = BBBWinRateEstimator(context_dim)
            self.prior_var = prior_var
        elif method=='MC Dropout':
            self.winrate_model = MCDropoutWinRateEstimator(context_dim)
        elif method=='NoisyNet':
            self.winrate_model = NoisyNetWinRateEstimator(context_dim)
        self.model_initialised = False
        super(ValueSamplingBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # If it is the first iteration, collect data with simple initial policy
            gamma = self.rng.normal(self.gamma_mu, self.gamma_sigma)
        else:
            # Grid search over gamma
            n_values_search = 128
            gamma_grid = np.linspace(0.1, 1 ,n_values_search)
            x = torch.Tensor(np.hstack((np.tile(context, ((n_values_search, 1))), np.tile(estimated_CTR, (n_values_search, 1)),
                                        np.tile(value, (n_values_search, 1)), gamma_grid.reshape(-1,1))))

            if self.method=='Bayes by Backprop' or self.method=='NoisyNet':
                prob_win = self.winrate_model(x, True).detach().numpy().ravel()
            elif self.method=='MC Dropout':
                with torch.no_grad():
                    prob_win = self.winrate_model(x).detach().numpy().ravel()
            else:
                prob_win = self.winrate_model(x).detach().numpy().ravel()

            # U = W (V - P)
            expected_value = bid
            shaded_bids = expected_value * gamma_grid
            estimated_utility = prob_win * (expected_value - shaded_bids)
            gamma = gamma_grid[np.argmax(estimated_utility)]
        
        if self.method=='Epsilon-greedy' and self.rng.random()<self.eps:
            gamma = self.rng.uniform(0.1,1)
        elif self.method=='Gaussian Noise':
            gamma = np.clip(gamma+self.rng.normal(0,self.noise), 0.1, 1)

        bid *= gamma
        self.gammas.append(gamma)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # FALLBACK: if you lost every auction you participated in, your model collapsed
        # Revert to not shading for 1 round, to collect data with informational value
        if not won_mask.astype(np.uint8).sum():
            self.model_initialised = False
            print(f'! Fallback for {name}')
            return

        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        utilities = torch.Tensor(utilities)

        # Augment data with samples: if you shade 100%, you will lose
        # If you won now, you would have also won if you bid higher
        X = np.hstack((contexts.reshape(-1,self.context_dim), estimated_CTRs.reshape(-1,1), values.reshape(-1,1), np.array(self.gammas).reshape(-1, 1)))

        X_aug_neg = X.copy()
        X_aug_neg[:, -1] = 0.0

        X_aug_pos = X[won_mask].copy()
        X_aug_pos[:, -1] = np.maximum(X_aug_pos[:, -1], 1.0)

        X = torch.Tensor(np.vstack((X, X_aug_neg)))

        y = won_mask.astype(np.float32).reshape(-1,1)
        y = torch.Tensor(np.concatenate((y, np.zeros_like(y))))

        # Training config
        self.winrate_model.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-7, factor=0.1, verbose=True)
        losses = []
        best_epoch, best_loss = -1, np.inf
        batch_size = 128
        batch_num = int(8192/128)

        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            epoch_loss = 0

            for i in range(batch_num):
                X_mini = X[i:i+batch_size]
                y_mini = y[i:i+batch_size]
                optimizer.zero_grad()

                if self.method=='Bayes by Backprop':
                    loss = self.winrate_model.loss(X_mini, y_mini, batch_size, 2, self.prior_var)/batch_num
                elif self.method=='NoisyNet':
                    loss = self.winrate_model.loss(X_mini, y_mini, 2)/batch_num
                else:
                    loss = self.winrate_model.loss(X_mini, y_mini)/batch_num
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

        self.winrate_model.eval()
        # fig, ax = plt.subplots()
        # plt.title(f'{name}')
        # plt.plot(losses, label=r'P(win|$gamma$,x)')
        # plt.ylabel('Loss')
        # plt.legend()
        # fig.set_tight_layout(True)
        # plt.show()

        # Predict Utility -- \hat{u}
        # orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), np.array(self.gammas).reshape(-1, 1))))
        # W = self.winrate_model(orig_features).squeeze().detach().numpy()
        # print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

        self.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
        else:
            self.gammas = self.gammas[-memory:]

class StochasticPolicyBidder(Bidder):

    def __init__(self, rng, gamma_mu, gamma_sigma, context_dim, learning_object, exploration_method, use_WIS=True):
        self.gamma_mu = gamma_mu
        self.gamma_sigma = gamma_sigma
        self.context_dim = context_dim + 1
        self.gammas = []
        self.propensities = []

        assert learning_object in ['IS', 'Actor-critic', 'DR']
        self.learning_object = learning_object
        # Do we need a value estimator?
        if self.learning_object=='Actor-critic' or self.learning_object=='DR':
            self.winrate_model = PyTorchWinRateEstimator()
        # Vanilla IS or weighted IS?
        if self.learning_object=='IS' or self.learning_object=='DR':
            self.use_WIS = use_WIS

        assert exploration_method in ['Entropy Regularization', 'NoisyNet', 'Bayes by Backprop', 'MC Dropout']
        self.exploration_method = exploration_method
        if self.exploration_method=='NoisyNet' or self.exploration_method=='Bayes by Backprop':
            self.bidding_policy = BayesianPolicy(context_dim)
        self.bidding_policy = BidShadingContextualBandit(loss='Doubly Robust', winrate_model=self.winrate_model)
        self.model_initialised = False
        super(StochasticPolicyBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shading factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.prev_gamma - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
        else:
            # Option 2:
            # Sample from the contextual bandit
            x = torch.Tensor([estimated_CTR, value])
            with torch.no_grad():
                gamma, propensity = self.bidding_policy(x)
                gamma = torch.clip(gamma, 0.0, 1.0)

        bid *= gamma.detach().item() if self.model_initialised else gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        # utilities = torch.Tensor(utilities)

        ##############################
        # 1. TRAIN UTILITY ESTIMATOR #
        ##############################
        gammas_numpy = np.array([g.detach().item() if self.model_initialised else g for g in self.gammas])
        if self.model_initialised:
            # Predict Utility -- \hat{u}
            orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1))))
            W = self.winrate_model(orig_features).squeeze().detach().numpy()
            print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

            V = estimated_CTRs * values
            P = estimated_CTRs * values * gammas_numpy
            estimated_utilities = W * (V - P)

            errors = estimated_utilities - utilities
            print('Estimated Utility\t Mean Error:\t\t\t', errors.mean())
            print('Estimated Utility\t Mean Absolute Error:\t', np.abs(errors).mean())

        # Augment data with samples: if you shade 100%, you will lose
        # If you won now, you would have also won if you bid higher
        X = np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1)))

        X_aug_neg = X.copy()
        X_aug_neg[:, -1] = 0.0

        X_aug_pos = X[won_mask].copy()
        X_aug_pos[:, -1] = np.maximum(X_aug_pos[:, -1], 1.0)

        X = torch.Tensor(np.vstack((X, X_aug_neg)))

        y = won_mask.astype(np.uint8).reshape(-1,1)
        y = torch.Tensor(np.concatenate((y, np.zeros_like(y))))

        # Fit the model
        self.winrate_model.train()
        epochs = 8192 * 4
        lr = 3e-3
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=256, min_lr=1e-7, factor=0.2, verbose=True)
        criterion = torch.nn.BCELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()
            pred_y = self.winrate_model(X)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 1024:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)

        self.winrate_model.eval()

        # Predict Utility -- \hat{u}
        orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1))))
        W = self.winrate_model(orig_features).squeeze().detach().numpy()
        print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

        V = estimated_CTRs * values
        P = estimated_CTRs * values * gammas_numpy
        estimated_utilities = W * (V - P)

        errors = estimated_utilities - utilities
        print('Estimated Utility\t Mean Error:\t\t\t', errors.mean())
        print('Estimated Utility\t Mean Absolute Error:\t', np.abs(errors).mean())

        ##############################
        # 2. TRAIN DOUBLY ROBUST POLICY #
        ##############################
        utilities = torch.Tensor(utilities)
        estimated_utilities = torch.Tensor(estimated_utilities)
        gammas = torch.Tensor(self.gammas)

        # Prepare features
        X = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1))))

        if not self.model_initialised:
            self.bidding_policy.initialise_policy(X, gammas)

        # Ensure we don't have propensities that are rounded to zero
        propensities = torch.clip(torch.Tensor(self.propensities), min=1e-15)

        # Fit the model
        self.bidding_policy.train()
        epochs = 8192 * 4
        lr = 7e-3
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, threshold=5e-3, verbose=True)

        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()  # Setting our stored gradients equal to zero
            loss = self.bidding_policy.loss(X, gammas, propensities, utilities, utility_estimates=estimated_utilities, winrate_model=self.winrate_model, importance_weight_clipping_eps=50.0)
            loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
            optimizer.step()  # Updates weights and biases with the optimizer (SGD)
            losses.append(loss.item())
            scheduler.step(loss)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 512:
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

        pred_gammas, _ = self.bidding_policy(X)
        pred_gammas = pred_gammas.detach().numpy()
        print(name, 'Number of samples: ', X.shape)
        print(name, 'Predicted Gammas: ', pred_gammas.min(), pred_gammas.max(), pred_gammas.mean())

        self.model_initialised = True
        self.bidding_policy.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]
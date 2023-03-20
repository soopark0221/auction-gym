import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numba import jit
from scipy.optimize import minimize
from torch.nn import functional as F
from tqdm import tqdm


@jit(nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input, sample=True):
        if self.training or sample:
            weight_sigma = torch.log1p(self.weight_rho)
            bias_sigma = torch.log1p(self.bias_rho)
            weight = self.weight_mu + weight_sigma * torch.randn(self.weight_mu.size())
            bias = self.bias_mu + bias_sigma * torch.randn(self.bias_mu.size())
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)
    
    def KL_div(self, prior_var):
        weight_sigma = torch.log1p(self.weight_rho)
        bias_sigma = torch.log1p(self.bias_rho)
        d = self.weight_mu.nelement() + self.bias_mu.nelement()
        return 0.5*(-torch.log(weight_sigma).sum() -torch.log(bias_sigma).sum() + (torch.sum(weight_sigma)+torch.sum(bias_sigma)) / prior_var \
                + (torch.sum(self.weight_mu**2) + torch.sum(self.bias_mu**2)) / prior_var - d + d*np.log(prior_var))



# This is an implementation of Algorithm 3 (Regularised Bayesian Logistic Regression with a Laplace Approximation)
# from "An Empirical Evaluation of Thompson Sampling" by Olivier Chapelle & Lihong Li
# https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf

class PyTorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_dim, n_items):
        super(PyTorchLogisticRegression, self).__init__()
        self.m = torch.nn.Parameter(torch.Tensor(n_items, n_dim + 1))
        torch.nn.init.normal_(self.m, mean=0.0, std=1.0)
        self.prev_iter_m = self.m.detach().clone()
        self.q = torch.ones((n_items, n_dim + 1))
        self.logloss = torch.nn.BCELoss(reduction='sum')
        self.eval()

    def forward(self, x, sample=False):
        ''' Predict outcome for all items, allow for posterior sampling '''
        if sample:
            return torch.sigmoid(F.linear(x, self.m + torch.normal(mean=0.0, std=1.0/torch.sqrt(self.q))))
        else:
            return torch.sigmoid(F.linear(x, self.m))

    def predict_item(self, x, a):
        ''' Predict outcome for an item a, only MAP '''
        return torch.sigmoid((x * self.m[a]).sum(axis=1))

    def loss(self, predictions, labels):
        prior_dist = self.q[:, :-1] * (self.prev_iter_m[:, :-1] - self.m[:, :-1])**2
        return 0.5 * prior_dist.sum() + self.logloss(predictions, labels)

    def laplace_approx(self, X, item):
        P = (1 + torch.exp(1 - X.matmul(self.m[item, :].T))) ** (-1)
        self.q[item, :] += (P*(1-P)).T.matmul(X ** 2).squeeze(0)

    def update_prior(self):
        self.prev_iter_m = self.m.detach().clone()


class PyTorchWinRateEstimator(torch.nn.Module):
    def __init__(self, context_dim):
        super(PyTorchWinRateEstimator, self).__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(context_dim+4, 1, bias=True),
            torch.nn.Sigmoid()
        )
        self.metric = nn.BCELoss()
        self.eval()

    def forward(self, x):
        return self.ffn(x)
    
    def loss(self, x, y):
        return self.metric(self.ffn(x), y)
    
    
class NeuralWinRateEstimator(nn.Module):
    def __init__(self, context_dim):
        super(NeuralWinRateEstimator, self).__init__()
        self.ffn = nn.Sequential(*[
            nn.Linear(context_dim+4,16),
            nn.ReLU(),
            nn.Linear(16,1)]
            )
        self.metric = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=False):
        return torch.sigmoid(self.ffn(x))
    
    def loss(self, x, y):
        logits = self.ffn(x)
        return self.metric(logits, y)
    
class BBBWinRateEstimator(nn.Module):
    def __init__(self, context_dim):
        super(BBBWinRateEstimator, self).__init__()
        self.linear1 = BayesianLinear(context_dim+4,8)
        self.relu = nn.ReLU()
        self.linear2 = BayesianLinear(8,1)
        self.metric = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=True):
        x = self.relu(self.linear1(x,sample))
        return torch.sigmoid(self.linear2(x, sample))
    
    def loss(self, x, y, N, sample_num, prior_var):
        loss = self.linear1.KL_div(prior_var)/N + self.linear2.KL_div(prior_var)/N
        for _ in range(sample_num):
            logits = self.linear2(self.relu(self.linear1(x)))
            loss += self.metric(logits.squeeze(), y.squeeze())/sample_num
        return loss
    
class MCDropoutWinRateEstimator(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.ffn = nn.Sequential(*[
            nn.Linear(context_dim+4,8),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(8,1)])
        self.metric = nn.BCEWithLogitsLoss()
        self.train()
    
    def forward(self, x):
        return torch.sigmoid(self.ffn(x))
    
    def loss(self, x, y):
        logits = self.ffn(x)
        return self.metric(logits, y)

class NoisyNetWinRateEstimator(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.linear1 = BayesianLinear(context_dim+4,8)
        self.relu = nn.ReLU()
        self.linear2 = BayesianLinear(8,1)
        self.metric = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=False):
        x = self.relu(self.linear1(x,sample))
        return torch.sigmoid(self.linear2(x, sample))
    
    def loss(self, x, y, sample_num):
        loss = 0
        for _ in range(sample_num):
            logits = self.linear2(self.relu(self.linear1(x)))
            loss += self.metric(logits.squeeze(), y.squeeze())/sample_num
        return loss


class BidShadingPolicy(torch.nn.Module):
    def __init__(self, context_dim):
        super(BidShadingPolicy, self).__init__()
        # Input: context, P(click), value
        # Output: mu, sigma for Gaussian bid shading distribution
        # Learnt to maximise E[P(win|gamma)*(value - price)] when gamma ~ N(mu, sigma)
        self.shared_linear = torch.nn.Linear(context_dim + 3, 2, bias=True)

        self.mu_linear_hidden = torch.nn.Linear(2, 2)
        self.mu_linear_out = torch.nn.Linear(2, 1)

        self.sigma_linear_hidden = torch.nn.Linear(2, 2)
        self.sigma_linear_out = torch.nn.Linear(2, 1)
        self.eval()

        self.min_sigma = 1e-2

    def forward(self, x):
        x = self.shared_linear(x)
        mu = torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(x)))
        sigma = torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(x))) + self.min_sigma
        dist = torch.distributions.normal.Normal(mu, sigma)
        sampled_value = dist.rsample()
        propensity = torch.exp(dist.log_prob(sampled_value))
        sampled_value = torch.clip(sampled_value, min=0.0, max=1.0)
        return sampled_value, propensity


class BidShadingContextualBandit(torch.nn.Module):
    def __init__(self, loss, context_dim):
        super(BidShadingContextualBandit, self).__init__()

        self.shared_linear = torch.nn.Linear(context_dim+3, 2, bias=True)

        self.mu_linear_out = torch.nn.Linear(2, 1)

        self.sigma_linear_out = torch.nn.Linear(2, 1)
        self.eval()

        self.min_sigma = 1e-2

        self.loss_name = loss

        self.model_initialised = False

    def initialise_policy(self, observed_contexts, observed_gammas):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        criterion = torch.nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        batch_size = 128
        batch_num = int(8192/128)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                contexts = observed_contexts[i:i+batch_num]
                gammas = observed_gammas[i:i+batch_num]
                optimizer.zero_grad()
                predicted_mu_gammas = torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(self.shared_linear(contexts))))
                predicted_sigma_gammas = torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(self.shared_linear(contexts))))
                loss = criterion(predicted_mu_gammas.squeeze(), observed_gammas)/batch_num + criterion(predicted_sigma_gammas.squeeze(), torch.ones_like(gammas) * .05)/batch_num
                loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
                optimizer.step()  # Updates weights and biases with the optimizer (SGD)
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 100:
                print(f'Stopping at Epoch {epoch}')
                break

        # fig, ax = plt.subplots()
        # plt.title(f'Initialising policy')
        # plt.plot(losses, label=r'Loss')
        # plt.ylabel('MSE with logging policy')
        # plt.legend()
        # fig.set_tight_layout(True)
        #plt.show()

        # print('Predicted mu Gammas: ', predicted_mu_gammas.min(), predicted_mu_gammas.max(), predicted_mu_gammas.mean())
        # print('Predicted sigma Gammas: ', predicted_sigma_gammas.min(), predicted_sigma_gammas.max(), predicted_sigma_gammas.mean())

    def forward(self, x):
        x = self.shared_linear(x)
        dist = torch.distributions.normal.Normal(
            torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(x))),
            torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(x))) + self.min_sigma
        )
        sampled_value = dist.rsample()
        propensity = torch.exp(dist.log_prob(sampled_value))
        sampled_value = torch.clip(sampled_value, min=0.0, max=1.0)
        return sampled_value, propensity

    def normal_pdf(self, x, gamma):
        # Get distribution over bid shading factors
        x = self.shared_linear(x)
        mu = torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(x)))
        sigma = torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(x))) + self.min_sigma
        mu = mu.squeeze()
        sigma = sigma.squeeze()
        # Compute the density for gamma under a Gaussian centered at mu -- prevent overflow
        return mu, sigma, torch.clip(torch.exp(-((mu - gamma) / sigma)**2/2) / (sigma * np.sqrt(2 * np.pi)), min=1e-30)

    def loss(self, observed_context, observed_gamma, logging_pp, utility, utility_estimates=None, winrate_model=None, KL_weight=5e-2, importance_weight_clipping_eps=np.inf):

        mean_gamma_target, sigma_gamma_target, target_pp = self.normal_pdf(observed_context, observed_gamma)

        # If not initialised, do a single round of on-policy REINFORCE
        # The issue is that without proper initialisation, propensities vanish
        if (self.loss_name == 'REINFORCE'): # or (not self.model_initialised)
            return (-target_pp * utility).mean()

        elif self.loss_name == 'REINFORCE_offpolicy':
            importance_weights = target_pp / logging_pp
            return (-importance_weights * utility).mean()

        elif self.loss_name == 'TRPO':
            # https://arxiv.org/abs/1502.05477
            importance_weights = target_pp / logging_pp
            expected_utility = torch.mean(importance_weights * utility)
            KLdiv = (sigma_gamma_target**2 + (mean_gamma_target - observed_gamma)**2) / (2 * sigma_gamma_target**2) - 0.5
            # Simpler proxy for KL divergence
            # KLdiv = (mean_gamma_target - observed_gamma)**2
            return - expected_utility + KLdiv.mean() * KL_weight

        elif self.loss_name == 'PPO':
            # https://arxiv.org/pdf/1707.06347.pdf
            # NOTE: clipping is actually proposed in an additive manner
            importance_weights = target_pp / logging_pp
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0/importance_weight_clipping_eps,
                                                    max=importance_weight_clipping_eps)
            return - torch.min(importance_weights * utility, clipped_importance_weights * utility).mean()

        elif self.loss_name == 'Doubly Robust':
            importance_weights = target_pp / logging_pp

            DR_IPS = (utility - utility_estimates) * torch.clip(importance_weights, min=1.0/importance_weight_clipping_eps, max=importance_weight_clipping_eps)

            dist = torch.distributions.normal.Normal(
                mean_gamma_target,
                sigma_gamma_target
            )

            sampled_gamma = torch.clip(dist.rsample(), min=0.0, max=1.0)
            features_for_p_win = torch.hstack((observed_context, sampled_gamma.reshape(-1,1)))

            W = winrate_model(features_for_p_win).squeeze()

            V = observed_context[:,0].squeeze() * observed_context[:,1].squeeze()
            P = observed_context[:,0].squeeze() * observed_context[:,1].squeeze() * sampled_gamma

            DR_DM = W * (V - P)

            return -(DR_IPS + DR_DM).mean()

class BayesianStochasticPolicy(nn.Module):
    def __init__(self, context_dim, loss_type):
        super().__init__()

        self.base = BayesianLinear(context_dim+3, 8)
        self.mu_head = BayesianLinear(8, 1)
        self.sigma_head = BayesianLinear(8,1)
        self.eval()

        self.min_sigma = 1e-2
        self.loss_type = loss_type
        self.model_initialised = False

    def initialise_policy(self, contexts, gammas):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        metric = nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = 8192
        B = 128
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = contexts[i:i+B]
                gamma_mini = gammas[i:i+B]
                optimizer.zero_grad()
                gamma_mu = F.sigmoid(self.mu_head(F.relu(self.base(context_mini))))
                gamma_sigma = F.sigmoid(self.sigma_head(F.relu(self.base(context_mini))))
                loss = metric(gamma_mu.squeeze(), gamma_mini)+ metric(gamma_sigma.squeeze(), torch.ones_like(gammas) * 0.1)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 100:
                print(f'Stopping at Epoch {epoch}')
                break

    def forward(self, x, MAP_propensity):
        x = F.relu(self.base(x))
        mu = F.sigmoid(self.mu_head(x)).squeeze()
        sigma = F.sigmoid(self.sigma_head(x)).squeeze() + self.min_sigma
        dist = torch.distributions.normal.Normal(mu, sigma)
        gamma = dist.rsample()
        if MAP_propensity:
            x_MAP = F.relu(self.base(x, False))
            mu_MAP = F.sigmoid(self.mu_head(x_MAP, False)).squeeze()
            sigma_MAP = F.sigmoid(self.sigma_head(x_MAP, False)).squeeze() + self.min_sigma
            dist_MAP = torch.distributions.normal.Normal(mu_MAP, sigma_MAP)
            propensity = torch.exp(dist_MAP.log_prob(gamma))
        else:
            propensity = torch.exp(dist.log_prob(gamma))
        gamma = torch.clip(gamma, min=0.0, max=1.0)
        return gamma, propensity

    def normal_pdf(self, x, gamma):
        x = F.relu(self.base(x))
        mu = F.sigmoid(self.mu_head(x)).squeeze()
        sigma = F.sigmoid(self.sigma_head(x)).squeeze() + self.min_sigma
        dist = torch.distributions.Normal(mu, sigma)
        return torch.exp(dist.log_prob(gamma))

    def loss(self, context, gamma, logging_pp, utility,
             winrate_model=None, use_WIS=True, policy_clipping=0.5):
        context = torch.Tensor(context)
        gamma = torch.Tensor(gamma)
        target_pp = self.normal_pdf(context, gamma)

        if self.loss_type == 'REINFORCE': # vanilla off-policy REINFORCE
            importance_weights = target_pp / logging_pp
            if use_WIS:
                importance_weights = importance_weights / torch.sum(importance_weights)
            loss =  (-importance_weights * utility).mean()
        
        elif self.loss_type == 'Actor-Critic': # PPO Actor-Critic
            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            utility_estimates = winrate_model(x)
            loss = - utility_estimates * target_pp / logging_pp

        elif self.loss_type == 'PPO-MC': # PPO Monte Carlo
            importance_weights = target_pp / logging_pp
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0-policy_clipping,
                                                    max=1.0+policy_clipping)
            loss = - torch.min(importance_weights * utility, clipped_importance_weights * utility).mean()
        
        elif self.loss_type == 'PPO-AC': # PPO Actor-Critic
            importance_weights = target_pp / logging_pp
            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            utility_estimates = winrate_model(x)
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0-policy_clipping,
                                                    max=1.0+policy_clipping)
            loss = - torch.min(importance_weights * utility_estimates, clipped_importance_weights * utility_estimates).mean()

        elif self.loss_type == 'Doubly Robust':
            importance_weights = target_pp / logging_pp

            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            q_estimate = winrate_model(x)

            IS_term = (utility - q_estimate) * torch.clip(importance_weights, min=1.0-policy_clipping, max=1.0+policy_clipping)

            # Monte Carlo approximation of V(context)
            v_estimate = q_estimate

            loss = -(IS_term + v_estimate).mean()
        
        return loss

        
class StochasticPolicy(nn.Module):
    def __init__(self, context_dim, loss_type, dropout=None):
        super().__init__()
        self.dropout = dropout
        
        self.base = nn.Linear(context_dim+3, 8)
        self.mu_head = nn.Linear(8, 1)
        self.sigma_head = nn.Linear(8,1)
        if dropout is not None:
            self.dropout_mu = nn.Dropout(dropout)
            self.dropout_sigma = nn.Dropout(dropout)
            self.train()

        self.min_sigma = 1e-2
        self.loss_type = loss_type
        self.model_initialised = False
    
    def mu(self, x):
        x = F.relu(self.base(x))
        if self.dropout is None:
            return F.sigmoid(self.mu_head(x))
        else:
            return F.sigmoid(self.mu_head(self.dropout_mu(x)))
    
    def sigma(self, x):
        x = F.relu(self.base(x))
        if self.dropout is None:
            return F.sigmoid(self.sigma_head(x))
        else:
            return F.sigmoid(self.sigma_head(self.dropout_mu(x)))

    def initialise_policy(self, contexts, gammas):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        metric = nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = 8192
        B = 128
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = contexts[i:i+B]
                gamma_mini = gammas[i:i+B]
                optimizer.zero_grad()
                gamma_mu = self.mu(context_mini)
                gamma_sigma = self.sigma(context_mini)
                loss = metric(gamma_mu.squeeze(), gamma_mini)+ metric(gamma_sigma.squeeze(), torch.ones_like(gammas) * 0.1)
                loss.backward()
                optimizer.step()  
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 100:
                print(f'Stopping at Epoch {epoch}')
                break

    def forward(self, x):
        if self.dropout is not None:
            self.train()
        mu = self.mu(x)
        sigma = self.sigma(x)
        dist = torch.distributions.Normal(mu, sigma)
        gamma = dist.rsample()
        gamma = torch.clip(gamma, min=0.0, max=1.0)
        propensity = torch.exp(dist.log_prob(gamma))
        return gamma, propensity

    def normal_pdf(self, x, gamma):
        if self.dropout is not None:
            self.train()
        mu = self.mu(x)
        sigma = self.sigma(x)
        dist = torch.distributions.Normal(mu, sigma)
        return torch.exp(dist.log_prob(gamma))

    def loss(self, context, gamma, logging_pp, utility,
             winrate_model=None, use_WIS=True, policy_clipping=0.5, entropy_factor=0.1):
        context = torch.Tensor(context)
        gamma = torch.Tensor(gamma)
        target_pp = self.normal_pdf(context, gamma)

        if self.loss_type == 'REINFORCE': # vanilla off-policy REINFORCE
            importance_weights = target_pp / logging_pp
            if use_WIS:
                importance_weights = importance_weights / torch.sum(importance_weights)
            loss =  (-importance_weights * utility).mean()
        
        elif self.loss_type == 'Actor-Critic': # PPO Actor-Critic
            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            utility_estimates = winrate_model(x)
            loss = - utility_estimates * target_pp / logging_pp

        elif self.loss_type == 'PPO-MC': # PPO Monte Carlo
            importance_weights = target_pp / logging_pp
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0-policy_clipping,
                                                    max=1.0+policy_clipping)
            loss = - torch.min(importance_weights * utility, clipped_importance_weights * utility).mean()
        
        elif self.loss_type == 'PPO-AC': # PPO Actor-Critic
            importance_weights = target_pp / logging_pp
            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            utility_estimates = winrate_model(x)
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0-policy_clipping,
                                                    max=1.0+policy_clipping)
            loss = - torch.min(importance_weights * utility_estimates, clipped_importance_weights * utility_estimates).mean()

        elif self.loss_type == 'Doubly Robust':
            importance_weights = target_pp / logging_pp

            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            q_estimate = winrate_model(x)

            IS_term = (utility - q_estimate) * torch.clip(importance_weights, min=1.0-policy_clipping, max=1.0+policy_clipping)

            # Monte Carlo approximation of V(context)
            v_estimate = q_estimate

            loss = -(IS_term + v_estimate).mean()
        
        loss -= entropy_factor * self.entropy(context)
        
        return loss

    def entropy(self, context):
        context = F.relu(self.base(context))
        sigma = F.sigmoid(self.sigma_head(context)) + self.min_sigma
        return (1. + torch.log(2 * np.pi * sigma.squeeze())).mean()/2
          

class DeterministicPolicy(nn.Module):
    def __init__(self, context_dim):
        super().__init__()

        self.ffn = nn.Sequential(*[
            nn.Linear(context_dim+3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
            ])
        self.eval()

    def initialise_policy(self, context, gamma):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        metric = torch.nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = 8192
        B = 128
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = context[i:i+batch_num]
                gamma_mini = gamma[i:i+batch_num]
                optimizer.zero_grad()
                gamma_pred = self.ffn(context_mini)
                loss = metric(gamma_pred.squeeze(), gamma_mini)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 100:
                print(f'Stopping at Epoch {epoch}')
                break

    def forward(self, x):
        return self.ffn(x)
    
    def loss(self, winrate_model, context):
        gamma_pred = self(context).reshape(-1,1)
        utility = winrate_model(torch.cat([context, gamma_pred], dim=1))
        return torch.mean(utility)

class BayesianDeterministicPolicy(nn.Module):
    def __init__(self, context_dim, prior_var):
        super().__init__()

        self.linear1 = BayesianLinear(context_dim+3, 8)
        self.linear2 = BayesianLinear(8, 1)
        self.eval()

        self.prior_var = prior_var
        self.model_initialised = False

    def initialise_policy(self, context, gamma):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        metric = torch.nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = 8192
        B = 128
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = context[i:i+batch_num]
                gamma_mini = gamma[i:i+batch_num]
                optimizer.zero_grad()
                gamma_pred = F.relu(self.linear1(context_mini))
                gamma_pred = F.sigmoid(self.linear2(gamma_pred))
                loss = metric(gamma_pred.squeeze(), gamma_mini)
                loss += (self.linear1.KL_div(self.prior_var) + self.linear2.KL_div(self.prior_var))/N
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 100:
                print(f'Stopping at Epoch {epoch}')
                break

    def forward(self, x):
        x = F.relu(self.linear1(x, True))
        return F.sigmoid(self.linear2(x, True))
    
    def loss(self, winrate_model, context, N):
        kl_penalty = (self.linear1.KL_div(self.prior_var) + self.linear2.KL_div(self.prior_var))/N
        gamma_pred = self(context).reshape(-1,1)
        utility = winrate_model(torch.cat([context, gamma_pred], dim=1))
        return torch.mean(utility) + kl_penalty
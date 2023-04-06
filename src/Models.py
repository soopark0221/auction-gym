import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numba import jit
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_rho = nn.Parameter(torch.empty((out_features, in_features)))
        
        self.bias_mu = nn.Parameter(torch.empty((out_features,)))
        self.bias_rho = nn.Parameter(torch.empty((out_features,)))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu)
        nn.init.uniform_(self.weight_rho, -0.02, 0.02)
        nn.init.uniform_(self.bias_mu, -np.sqrt(3/self.weight_mu.size(0)), np.sqrt(3/self.weight_mu.size(0)))
        nn.init.uniform_(self.bias_rho, -0.02, 0.02)
        
    def forward(self, input, sample=True):
        if self.training or sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)
    
    def KL_div(self, prior_var):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        d = self.weight_mu.nelement() + self.bias_mu.nelement()
        return 0.5*(-torch.log(weight_sigma).sum() -torch.log(bias_sigma).sum() + (torch.sum(weight_sigma)+torch.sum(bias_sigma)) / prior_var \
                + (torch.sum(self.weight_mu**2) + torch.sum(self.bias_mu**2)) / prior_var - d + d*np.log(prior_var))
    
    def get_uncertainty(self):
        with torch.no_grad():
            weight_sigma = torch.log1p(torch.exp(self.weight_rho)).numpy(force=True)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho)).numpy(force=True)
        return np.concatenate([weight_sigma.reshape(-1), bias_sigma.reshape(-1)])


# This is an implementation of Algorithm 3 (Regularised Bayesian Logistic Regression with a Laplace Approximation)
# from "An Empirical Evaluation of Thompson Sampling" by Olivier Chapelle & Lihong Li
# https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf

class DiagLogisticRegression(torch.nn.Module):
    def __init__(self, n_dim, n_items):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m = torch.nn.Parameter(torch.Tensor(n_items, n_dim))
        torch.nn.init.normal_(self.m, mean=0.0, std=1.0)
        self.prev_iter_m = self.m.detach().clone().to(self.device)
        self.q = torch.ones((n_items, n_dim)).to(self.device)
        self.logloss = torch.nn.BCELoss(reduction='sum')
        self.eval()

    def forward(self, x, sample=False):
        ''' Predict outcome for all items, allow for posterior sampling '''
        if sample:
            return torch.sigmoid(F.linear(x, self.m + torch.normal(mean=0.0, std=1.0/torch.sqrt(self.q).to(self.device))))
        else:
            return torch.sigmoid(F.linear(x, self.m))

    def predict_item(self, x, a):
        ''' Predict outcome for an item a, only MAP '''
        return torch.sigmoid((x * self.m[a]).sum(axis=1))

    def loss(self, predictions, labels):
        prior_dist = self.q * (self.prev_iter_m - self.m)**2
        return 0.5 * prior_dist.sum() + self.logloss(predictions, labels)

    def laplace_approx(self, X, item):
        P = (1 + torch.exp(1 - X.matmul(self.m[item, :].T))) ** (-1)
        self.q[item, :] += (P*(1-P)).T.matmul(X ** 2).squeeze(0)

    def update_prior(self):
        self.prev_iter_m = self.m.detach().clone()
    
    def get_uncertainty(self):
        return self.q.numpy(force=True).reshape(-1)

class LinearRegression:
    def __init__(self,context_dim, num_items, mode, rng, c=2.):
        super().__init__()
        self.rng = rng
        self.mode = mode # UCB or TS    
        self.K = num_items
        self.d = context_dim

        self.N = np.zeros((self.K,), dtype=int)
        self.lambda0 = 1.0    # regularization constant
        self.c = c

        self.yt_y = np.zeros((self.K,))
        self.Xt_y = np.zeros((self.K, self.d))
        self.Xt_X = np.zeros((self.K, self.d, self.d))
        temp = [np.identity(self.d)/self.lambda0 for _ in range(self.K)]
        self.S = np.stack(temp)
        self.sqrt_S = self.S.copy()
        self.m = np.zeros((self.K, self.d))

    def update(self, contexts, items, outcomes):
        for k in range(self.K):
            mask = items==k
            X = contexts[mask][self.N[k]:].reshape(-1,self.d)
            y = outcomes[mask][self.N[k]:].reshape(-1)
            if y.shape[0]>0:
                self.N[k] += y.shape[0]

                self.yt_y[k] += np.dot(y,y)
                self.Xt_y[k,:] += np.matmul(X.T, y).reshape(-1)
                self.Xt_X[k,:,:] += np.matmul(X.T, X)

                self.S[k,:,:] = np.linalg.inv(self.Xt_X[k,:,:] + self.lambda0*np.identity(self.d))
                self.m[k,:] = self.S[k,:,:] @ self.Xt_y[k,:]
                D, V = np.linalg.eig(self.S[k,:,:])
                D, V = D.real, V.real
                self.sqrt_S[k,:,:] = V @ np.diag(np.sqrt(D))

    def estimate_CTR(self, context, UCB=False, TS=False):
        context = context.reshape(-1)
        if UCB:
            return self.m @ context + self.c * np.sqrt(np.tensordot(context.T,np.tensordot(self.S, context, axes=([2],[0])), axes=([0],[1])))
        elif TS:
            m = self.m.numpy(force=True) + self.sqrt_S @ self.rng.normal(0,1,self.d)
            return m @ context
        else:
            return self.m @ context

    def get_uncertainty(self):
        eigvals = [np.linalg.eigvals(self.S[k,:,:]).reshape(-1) for k in range(self.K)]
        return np.concatenate(eigvals).real

class LogisticRegression(nn.Module):
    def __init__(self, context_dim, num_items, mode, rng, c=1.0):
        super().__init__()
        self.rng = rng
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = num_items
        self.d = context_dim
        self.c = c

        self.m = nn.Parameter(torch.Tensor(self.K, self.d))
        nn.init.kaiming_uniform_(self.m)

        self.S0_inv = torch.Tensor(np.identity(self.d)).to(self.device)
        temp = [np.identity(self.d) for _ in range(self.K)]
        self.S_inv = np.stack(temp)
        self.sqrt_S = self.S_inv.copy()
        self.S = torch.Tensor(self.S_inv.copy()).to(self.device)

        self.BCE = torch.nn.BCELoss(reduction='sum')
    
    def forward(self, X, A):
        return torch.sigmoid(torch.sum(X * self.m[A], dim=1))
    
    def update(self, contexts, items, outcomes, name):
        X = torch.Tensor(contexts).to(self.device)
        A = torch.LongTensor(items).to(self.device)
        y = torch.Tensor(outcomes).to(self.device)

        epochs = 100
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []
        for epoch in range(int(epochs)):
                optimizer.zero_grad()
                loss = self.loss(X, A, y, self.S0_inv)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        
        y = self(X, A).numpy(force=True)
        X = contexts.reshape(-1,self.d)
        for k in range(self.K):
            mask = items==k
            y_ = y[mask]
            X_ = X[mask,:]
            N = y_.shape[0]
            y_ = y_ * (1 - y_)
            S_inv = self.S0_inv.numpy(force=True)
            for n in range(N):
                S_inv += y_[n] * X_[n,:].reshape(-1,1) @ X_[n,:].reshape(1,-1)
            self.S_inv[k,:,:] = S_inv

            S = np.linalg.inv(S_inv)
            D, V = np.linalg.eig(S)
            D, V = D.real, V.real
            self.sqrt_S[k,:,:] = V @ np.diag(np.sqrt(D))
            self.S[k,:,:] = torch.Tensor(S).to(self.device)
            
    def loss(self, X, A, y, S0_inv):
        y_pred = self(X, A)
        loss = self.BCE(y_pred, y)
        for k in range(self.K):
            loss += torch.sum(self.m[k,:] @ S0_inv @ self.m[k,:]/2)
        return loss
    
    def estimate_CTR(self, context, UCB=False, TS=False):
        X = torch.Tensor(context.reshape(-1)).to(self.device)
        if UCB:
            U = torch.sigmoid(torch.matmul(self.m, X) + self.c * torch.sqrt(torch.tensordot(X.T,torch.tensordot(self.S, X, dims=([2],[0])), dims=([0],[1]))))
            return U.numpy(force=True)
        elif TS:
            m = self.m.numpy(force=True) + self.sqrt_S @ self.rng.normal(0,1,self.d)
            return (1 + np.exp(m @ context))**(-1)
        else:
            return torch.sigmoid(torch.matmul(self.m, X)).numpy(force=True)
    
    def get_uncertainty(self):
        S_ = self.S.numpy(force=True)
        eigvals = [np.linalg.eigvals(S_[k,:,:]).reshape(-1) for k in range(self.K)]
        S_ += 1e-4 * np.identity(self.d)
        return np.concatenate(eigvals).real
       
class BayesianNeuralRegression(nn.Module):
    def __init__(self, n_dim, n_items, prior_var):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_dim = n_dim
        self.n_items = n_items
        self.prior_var = prior_var
        self.linear1 = BayesianLinear(n_dim, n_dim)
        self.linear2 = BayesianLinear(n_dim, n_items)
        self.criterion = nn.BCELoss()
        self.eval()

    def forward(self, x, sample=False):
        ''' Predict outcome for all items, allow for posterior sampling '''
        x = torch.relu(self.linear1(x, sample))
        return torch.sigmoid(self.linear2(x, sample))

    def predict_item(self, x, a):
        ''' Predict outcome for an item a, only MAP '''
        return self(x, True)[range(a.size(0)),a]

    def loss(self, predictions, labels, N):
        kl_div = self.linear1.KL_div(self.prior_var) + self.linear2.KL_div(self.prior_var)
        return self.criterion(predictions, labels) + kl_div / N
    
    def get_uncertainty(self):
        uncertainties = [self.linear1.get_uncertainty(),
                         self.linear2.get_uncertainty()]
        return np.concatenate(uncertainties)

class NeuralRegression(nn.Module):
    def __init__(self, n_dim, n_items):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_dim = n_dim
        self.n_items = n_items
        self.feature = nn.Linear(self.n_dim, 16)
        self.head = nn.Linear(16, self.n_items)
        self.BCE = nn.BCELoss()
        self.eval()

    def forward(self, x):
        x = torch.relu(self.feature(x))
        return torch.sigmoid(self.head(x))

    def predict_item(self, x, a):
        return self(x)[range(a.size(0)),a]

    def loss(self, predictions, labels):
        return self.BCE(predictions, labels)


class PyTorchWinRateEstimator(torch.nn.Module):
    def __init__(self, context_dim):
        super(PyTorchWinRateEstimator, self).__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(context_dim+3, 1, bias=True),
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
        super().__init__()
        self.ffn = nn.Sequential(*[
            nn.Linear(context_dim+1,16),
            nn.ReLU(),
            nn.Linear(16,1)]
            )
        self.BCE = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=False):
        return torch.sigmoid(self.ffn(x))
    
    def loss(self, x, y):
        logits = self.ffn(x)
        return self.BCE(logits, y)
    
class BBBWinRateEstimator(nn.Module):
    def __init__(self, context_dim):
        super(BBBWinRateEstimator, self).__init__()
        self.linear1 = BayesianLinear(context_dim+3,16)
        self.relu = nn.ReLU()
        self.linear2 = BayesianLinear(16,1)
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
    
    def get_uncertainty(self):
        uncertainties = [self.linear1.get_uncertainty(),
                         self.linear2.get_uncertainty()]
        return np.concatenate(uncertainties)
    
class MCDropoutWinRateEstimator(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.ffn = nn.Sequential(*[
            nn.Linear(context_dim+3,16),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(16,1)])
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
        self.linear1 = BayesianLinear(context_dim+3,16)
        self.relu = nn.ReLU()
        self.linear2 = BayesianLinear(16,1)
        self.metric = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=True):
        x = self.relu(self.linear1(x,sample))
        return torch.sigmoid(self.linear2(x, sample))
    
    def loss(self, x, y, sample_num):
        loss = 0
        for _ in range(sample_num):
            logits = self.linear2(self.relu(self.linear1(x)))
            loss += self.metric(logits.squeeze(), y.squeeze())/sample_num
        return loss
    
    def get_uncertainty(self):
        uncertainties = [self.linear1.get_uncertainty(),
                         self.linear2.get_uncertainty()]
        return np.concatenate(uncertainties)


class BayesianStochasticPolicy(nn.Module):
    def __init__(self, context_dim, loss_type, use_WIS=True):
        super().__init__()
        self.use_WIS = use_WIS

        self.base = BayesianLinear(context_dim+3, 16)
        self.mu_head = BayesianLinear(16, 1)
        self.sigma_head = BayesianLinear(16, 1)
        self.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_sigma = 1e-2
        self.loss_type = loss_type
        self.model_initialised = False

    def initialise_policy(self, contexts, gammas):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=256, min_lr=1e-7, factor=0.2, verbose=True)
        metric = nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = contexts.size()[0]
        B = N
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = contexts[i:i+B]
                gamma_mini = gammas[i:i+B]
                optimizer.zero_grad()
                gamma_mu = F.softplus(self.mu_head(torch.relu(self.base(context_mini))))
                gamma_sigma = torch.sigmoid(self.sigma_head(torch.relu(self.base(context_mini))))
                loss = metric(gamma_mu.squeeze(), gamma_mini) + 0.1 * metric(gamma_sigma.squeeze(), torch.ones_like(gamma_mini).to(self.device) * 0.5)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 2000:
                print(f'Stopping at Epoch {epoch}')
                break
        
        # fig, ax = plt.subplots()
        # plt.title(f'Initialising policy')
        # plt.plot(losses, label=r'Loss')
        # plt.ylabel('MSE with logging policy')
        # plt.legend()
        # fig.set_tight_layout(True)
        # plt.show()

        pred_gamma_mu = F.softplus(self.mu_head(torch.relu(self.base(context_mini)))).numpy(force=True)
        pred_gamma_sigma = torch.sigmoid(self.sigma_head(torch.relu(self.base(context_mini)))).numpy(force=True)
        print('Predicted mu Gammas: ', pred_gamma_mu.min(), pred_gamma_mu.max(), pred_gamma_mu.mean())
        print('Predicted sigma Gammas: ', pred_gamma_sigma.min(), pred_gamma_sigma.max(), pred_gamma_sigma.mean())


    def forward(self, x, MAP_propensity=True):
        hidden = torch.relu(self.base(x))
        mu = F.softplus(self.mu_head(hidden)).squeeze()
        sigma = torch.sigmoid(self.sigma_head(hidden)).squeeze() + self.min_sigma
        dist = torch.distributions.normal.Normal(mu, sigma)
        gamma = dist.rsample()
        if MAP_propensity:
            x_MAP = torch.relu(self.base(x, False))
            mu_MAP = F.softplus(self.mu_head(x_MAP, False)).squeeze()
            sigma_MAP = torch.sigmoid(self.sigma_head(x_MAP, False)).squeeze() + self.min_sigma
            dist_MAP = torch.distributions.normal.Normal(mu_MAP, sigma_MAP)
            propensity = torch.exp(dist_MAP.log_prob(gamma))
        else:
            propensity = torch.exp(dist.log_prob(gamma))
        gamma = torch.clip(gamma, min=0.0, max=1.5)
        return gamma, propensity.numpy(force=True).item(), sigma.numpy(force=True)

    def normal_pdf(self, x, gamma):
        x = torch.relu(self.base(x))
        mu = F.softplus(self.mu_head(x)).squeeze()
        sigma = torch.sigmoid(self.sigma_head(x)).squeeze() + self.min_sigma
        dist = torch.distributions.Normal(mu, sigma)
        return dist, torch.exp(dist.log_prob(gamma))

    def loss(self, context, value, price, gamma, logging_pp, utility,
             winrate_model=None, policy_clipping=0.5):
        dist, target_pp = self.normal_pdf(context, gamma)

        if self.loss_type == 'REINFORCE': # vanilla off-policy REINFORCE
            importance_weights = torch.clip(target_pp / logging_pp, 0.01)
            if self.use_WIS:
                importance_weights = importance_weights / torch.sum(importance_weights)
            loss =  (-importance_weights * utility).mean()
        
        elif self.loss_type == 'Actor-Critic': # PPO Actor-Critic
            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            winrate = winrate_model(x)
            utility_estimates = winrate * (value - gamma * price)
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
            winrate = winrate_model(x)
            utility_estimates = winrate * (value - gamma * price)
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0-policy_clipping,
                                                    max=1.0+policy_clipping)
            loss = - torch.min(importance_weights * utility_estimates, clipped_importance_weights * utility_estimates).mean()

        elif self.loss_type == 'DR':
            importance_weights = target_pp / logging_pp

            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            winrate = winrate_model(x)
            q_estimate = winrate * (value - gamma * price)

            IS_clip = (utility - q_estimate) * torch.clip(importance_weights, min=1.0-policy_clipping, max=1.0+policy_clipping)
            IS = (utility - q_estimate) * importance_weights
            ppo_object = torch.min(IS, IS_clip)

            # Monte Carlo approximation of V(context)
            with torch.no_grad():
                sampled_gamma = dist.sample()
                x = torch.cat([context, sampled_gamma.reshape(-1,1)], dim=1)
                v_estimate = winrate_model(x) * (value - sampled_gamma * value)

            loss = -(ppo_object + v_estimate).mean()
        
        return loss
    
    def get_uncertainty(self):
        uncertainties = [self.base.get_uncertainty(),
                         self.mu_head.get_uncertainty(),
                         self.sigma_head.get_uncertainty()]
        return np.concatenate(uncertainties)
        
class StochasticPolicy(nn.Module):
    def __init__(self, context_dim, loss_type, dropout=None, use_WIS=True, entropy_factor=None):
        super().__init__()
        self.dropout = dropout
        self.use_WIS = use_WIS
        self.entropy_factor = entropy_factor
        
        self.base = nn.Linear(context_dim+3, 8)
        self.mu_head = nn.Linear(8, 1)
        self.sigma_head = nn.Linear(8,1)
        if dropout is not None:
            self.dropout_mu = nn.Dropout(dropout)
            self.dropout_sigma = nn.Dropout(dropout)
            self.train()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_sigma = 1e-2
        self.loss_type = loss_type
        self.model_initialised = False
    
    def mu(self, x):
        x = torch.relu(self.base(x))
        if self.dropout is None:
            return F.softplus(self.mu_head(x))
        else:
            return F.softplus(self.mu_head(self.dropout_mu(x)))
    
    def sigma(self, x):
        x = torch.relu(self.base(x))
        if self.dropout is None:
            return torch.sigmoid(self.sigma_head(x))
        else:
            return torch.sigmoid(self.sigma_head(self.dropout_mu(x)))

    def initialise_policy(self, contexts, gammas):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        metric = nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = contexts.size()[0]
        B = N
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = contexts[i:i+B]
                gamma_mini = gammas[i:i+B]
                optimizer.zero_grad()
                gamma_mu = self.mu(context_mini)
                gamma_sigma = self.sigma(context_mini)
                loss = metric(gamma_mu.squeeze(), gamma_mini)+ metric(gamma_sigma.squeeze(), torch.ones_like(gamma_mini).to(self.device) * 0.1)
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

        # fig, ax = plt.subplots()
        # plt.title(f'Initialising policy')
        # plt.plot(losses, label=r'Loss')
        # plt.ylabel('MSE with logging policy')
        # plt.legend()
        # fig.set_tight_layout(True)
        # plt.show()

        pred_gamma_mu = F.softplus(self.mu_head(torch.relu(self.base(context_mini)))).numpy(force=True)
        pred_gamma_sigma = F.softplus(self.sigma_head(torch.relu(self.base(context_mini)))).numpy(force=True)
        print('Predicted mu Gammas: ', pred_gamma_mu.min(), pred_gamma_mu.max(), pred_gamma_mu.mean())
        print('Predicted sigma Gammas: ', pred_gamma_sigma.min(), pred_gamma_sigma.max(), pred_gamma_sigma.mean())

    def forward(self, x):
        if self.dropout is not None:
            self.train()
        mu = self.mu(x)
        sigma = self.sigma(x)
        dist = torch.distributions.Normal(mu, sigma)
        gamma = dist.rsample()
        gamma = torch.clip(gamma, min=0.0, max=1.5)
        propensity = torch.exp(dist.log_prob(gamma))
        return gamma, propensity.numpy(force=True), sigma.numpy(force=True)

    def normal_pdf(self, x, gamma):
        if self.dropout is not None:
            self.train()
        mu = self.mu(x)
        sigma = self.sigma(x)
        dist = torch.distributions.Normal(mu, sigma)
        return dist, torch.exp(dist.log_prob(gamma))

    def loss(self, context, value, price, gamma, logging_pp, utility,
             winrate_model=None, policy_clipping=0.5):
        dist, target_pp = self.normal_pdf(context, gamma)

        if self.loss_type == 'REINFORCE': # vanilla off-policy REINFORCE
            importance_weights = torch.clip(target_pp / logging_pp, 0.01)
            if self.use_WIS:
                importance_weights = importance_weights / torch.sum(importance_weights)
            loss =  (-importance_weights * utility).mean()
        
        elif self.loss_type == 'Actor-Critic': # PPO Actor-Critic
            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            winrate = winrate_model(x)
            utility_estimates = winrate * (value - gamma * price)
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
            winrate = winrate_model(x)
            utility_estimates = winrate * (value - gamma * price)
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0-policy_clipping,
                                                    max=1.0+policy_clipping)
            loss = - torch.min(importance_weights * utility_estimates, clipped_importance_weights * utility_estimates).mean()

        elif self.loss_type == 'DR':
            importance_weights = target_pp / logging_pp

            x = torch.cat([context, gamma.reshape(-1,1)], dim=1)
            winrate = winrate_model(x)
            q_estimate = winrate * (value - gamma * price)

            IS_clip = (utility - q_estimate) * torch.clip(importance_weights, min=1.0-policy_clipping, max=1.0+policy_clipping)
            IS = (utility - q_estimate) * importance_weights
            ppo_object = torch.min(IS, IS_clip)

            # Monte Carlo approximation of V(context)
            with torch.no_grad():
                sampled_gamma = dist.rsample()
                x = torch.cat([context, sampled_gamma.reshape(-1,1)], dim=1)
                v_estimate = winrate_model(x) * (value - sampled_gamma * value)

            loss = -(ppo_object + v_estimate).mean()
        
        loss -= self.entropy_factor * self.entropy(context)
        
        return loss

    def entropy(self, context):
        context = torch.relu(self.base(context))
        sigma = F.softplus(self.sigma_head(context)) + self.min_sigma
        return (1. + torch.log(2 * np.pi * sigma.squeeze())).mean()/2
          

class DeterministicPolicy(nn.Module):
    def __init__(self, context_dim):
        super().__init__()

        self.ffn = nn.Sequential(*[
            nn.Linear(context_dim+1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
            ])
        self.eval()

    def initialise_policy(self, context, gamma):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        metric = torch.nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = context.size()[0]
        B = min(4096, N)
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = context[i:i+B]
                gamma_mini = gamma[i:i+B]
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
            elif epoch - best_epoch > 500:
                print(f'Stopping at Epoch {epoch}')
                break

    def forward(self, x):
        return torch.clip(self.ffn(x), 0.1, 1.5)
    
    def loss(self, winrate_model, X, V):
        gamma_pred = self(torch.cat([X, V], dim=1)).reshape(-1,1)
        winrate = winrate_model(torch.cat([X, V*gamma_pred], dim=1))
        utility = winrate * V * (1 - gamma_pred)
        return -torch.mean(utility)

class BayesianDeterministicPolicy(nn.Module):
    def __init__(self, context_dim, prior_var=None):
        super().__init__()

        self.linear1 = BayesianLinear(context_dim+1, 16)
        self.linear2 = BayesianLinear(16, 1)
        self.eval()

        self.prior_var = prior_var # if this is None, the bidder's exploring via NoisyNet
        self.model_initialised = False

    def initialise_policy(self, context, gamma):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        metric = torch.nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        N = context.size()[0]
        B = min(8192, N)
        batch_num = int(N/B)

        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            epoch_loss = 0
            for i in range(batch_num):
                context_mini = context[i:i+B]
                gamma_mini = gamma[i:i+B]
                optimizer.zero_grad()
                gamma_pred = torch.relu(self.linear1(context_mini))
                gamma_pred = F.softplus(self.linear2(gamma_pred))
                loss = metric(gamma_pred.squeeze(), gamma_mini)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
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

    def forward(self, x, sampling=True):
        x = torch.relu(self.linear1(x, sampling))
        return torch.clip(F.softplus(self.linear2(x, sampling)), 0.1, 1.5)
    
    def loss(self, winrate_model, X, V, N=None):
        gamma_pred = self(torch.cat([X, V], dim=1), True).reshape(-1,1)
        winrate = winrate_model(torch.cat([X, V*gamma_pred], dim=1))
        utility = winrate * V * (1 - gamma_pred)
        if self.prior_var is None:
            return -torch.mean(utility)
        else:
            kl_penalty = (self.linear1.KL_div(self.prior_var) + self.linear2.KL_div(self.prior_var))/N
            return -torch.mean(utility) + kl_penalty
    
    def get_uncertainty(self):
        uncertainties = [self.linear1.get_uncertainty(),
                         self.linear2.get_uncertainty()]
        return np.concatenate(uncertainties)
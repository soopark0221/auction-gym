import numpy as np
import torch
from torch import device
import torch.nn as nn
from numba import jit
from torch.nn import functional as F
from tqdm import tqdm

@jit(nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class NeuralRegression(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = input_dim
        self.h = latent_dim
        self.feature = nn.Linear(self.d, self.h)
        self.head = nn.Linear(self.h, 1)
        self.BCE = nn.BCELoss()
        self.eval()

    def initialize_weights(self):
        first_init=np.sqrt(4/self.H)*torch.randn((self.H,(self.d//2)-1)).to(self.device)
        first_init=torch.cat([first_init,torch.zeros(self.H,1).to(self.device),torch.zeros(self.H,1).to(self.device),first_init],axis=1)
        self.feature.weight.data=first_init 
        last_init=np.sqrt(2/self.K)*torch.randn((self.K,self.H//2)).to(self.device)
        self.head.weight.data=torch.cat([last_init,-last_init],axis=1)
        
    def forward(self, x):
        x = torch.relu(self.feature(x))
        return torch.sigmoid(self.head(x))

    def loss(self, predictions, labels):
        return self.BCE(predictions, labels)

class MultiheadNeuralRegression(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = input_dim
        self.h = latent_dim
        self.num_heads = num_heads
        self.linear = nn.Linear(self.d, self.h)
        self.heads = [nn.Linear(self.h, 1).to(self.device) for _ in range(num_heads)]
        self.BCE = nn.BCELoss()
        self.eval()
    
    def reset(self):
        self.linear = nn.Linear(self.d, self.h)
        self.heads = [nn.Linear(self.h, 1).to(self.device) for _ in range(self.num_heads)]
        self.to(self.device)
    
    def forward(self, x, i):
        x = torch.relu(self.linear(x))
        return torch.sigmoid(self.heads[i](x))

    def UCB_inference(self, x):
        x = torch.relu(self.linear(x))
        y = [torch.sigmoid(self.heads[i](x)).numpy(force=True).reshape(-1) for i in range(self.num_heads)]
        y = np.stack(y)
        return np.mean(y, axis=0), np.std(y, axis=0)

    def TS_inference(self,x):
        x = torch.relu(self.linear(x))
        y = [torch.sigmoid(self.heads[i](x)).numpy(force=True).reshape(-1) for i in range(self.num_heads)]
        return np.stack(y)

    def loss(self, x, y, i):
        return self.BCE(self(x,i).squeeze(), y)
        
    
class NeuralWinRateEstimator(nn.Module):
    def __init__(self, context_dim, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.H = 16
        if self.skip_connection:
            self.linear1 = nn.Linear(context_dim, self.H-1)
        else:
            self.linear1 = nn.Linear(context_dim+1, self.H)
        self.linear2 = nn.Linear(self.H, 1)
        self.BCE = nn.BCEWithLogitsLoss()
        self.eval()

    def forward(self, x, sample=False):
        if self.skip_connection:
            context = x[:,:-1]
            gamma = x[:,-1].reshape(-1,1)
            hidden = torch.sigmoid(self.linear1(context))
            hidden_ = torch.concat([hidden, gamma], dim=-1)
            return torch.sigmoid(self.linear2(hidden_))
        else:
            hidden = torch.sigmoid(self.linear1(x))
            return torch.sigmoid(self.linear2(hidden))
    
    def loss(self, x, y):
        if self.skip_connection:
            context = x[:,:-1]
            gamma = x[:,-1].reshape(-1,1)
            hidden = torch.sigmoid(self.linear1(context))
            hidden_ = torch.concat([hidden, gamma], dim=-1)
            logit = self.linear2(hidden_)
        else:
            hidden = torch.sigmoid(self.linear1(x))
            logit = self.linear2(hidden)
        return self.BCE(logit, y)
    

class StochasticPolicy(nn.Module):
    def __init__(self, context_dim, loss_type, use_WIS=True, entropy_factor=None, weight_clip=None):
        super().__init__()
        self.use_WIS = use_WIS
        self.entropy_factor = entropy_factor
        self.weight_clip = weight_clip
        
        self.base = nn.Linear(context_dim + 1, 16)
        self.mu_head = nn.Linear(16, 1)
        self.sigma_head = nn.Linear(16,1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_sigma = 1e-2
        self.loss_type = loss_type
        if loss_type=='Cloning':
            self.MSE = nn.MSELoss()
    
    def mu(self, x):
        x = torch.relu(self.base(x))
        return F.softplus(self.mu_head(x))
    
    def sigma(self, x):
        x = torch.relu(self.base(x))
        return F.softplus(self.sigma_head(x)) + self.min_sigma
    
    def forward(self, x):
        mu = self.mu(x)
        sigma = self.sigma(x)
        dist = torch.distributions.Normal(mu, sigma)
        b = dist.rsample()
        return b

    def normal_pdf(self, x, v, bid):
        x = torch.cat([x, v.reshape(-1,1)], dim=1)
        mu = self.mu(x)
        sigma = self.sigma(x)
        dist = torch.distributions.Normal(mu, sigma)
        return torch.exp(dist.log_prob(bid.reshape(-1,1)))

    def loss(self, context, estimated_value, bid, logging_pp=None, utility=None, winrate_model=None):
        input_policy = torch.cat([context, estimated_value.reshape(-1,1)], dim=1)
        if self.loss_type=='Cloning':
            b_pred = self(input_policy).squeeze()
            loss = self.MSE(b_pred, bid)

        if self.loss_type == 'REINFORCE':
            target_pp = self.normal_pdf(context, estimated_value, bid).reshape(-1)
            importance_weights = torch.clip(target_pp/(logging_pp+1e-6), 1/self.weight_clip, self.weight_clip)
            if self.use_WIS:
                N = context.size(0)
                loss = torch.tensor(0, dtype=float).to(self.device)
                for i in range(int(N/100)):
                    weighted_IW = importance_weights[i*100:(i+1)*100] / torch.sum(importance_weights[i*100:(i+1)*100])
                    loss += torch.sum(weighted_IW * utility[i*100:(i+1)*100])
                loss /= N
            else:
                loss = torch.mean(-importance_weights * utility)
            
        elif self.loss_type == 'Actor-Critic':
            target_pp = self.normal_pdf(context, estimated_value, bid).reshape(-1)
            input_winrate = torch.cat([context, bid.reshape(-1,1)], dim=1)

            winrate = winrate_model(input_winrate)
            utility_estimates = winrate * (estimated_value - bid)
            importance_weights = torch.clip(target_pp/(logging_pp+1e-6), 1/self.weight_clip, self.weight_clip)
            loss = - importance_weights * utility_estimates.reshape(-1)

        elif self.loss_type == 'DR':
            target_pp = self.normal_pdf(context, estimated_value, bid).reshape(-1)
            importance_weights = target_pp/(logging_pp+1e-6)

            input_winrate = torch.cat([context, bid.reshape(-1,1)], dim=1)
            winrate = winrate_model(input_winrate).reshape(-1)
            q_estimate = winrate * (estimated_value - bid)

            IS = (utility - q_estimate) * torch.clip(importance_weights, 1/self.weight_clip, self.weight_clip)

            # Monte Carlo approximation of V(context)
            v_estimate = winrate_model(input_winrate).reshape(-1) * (estimated_value - bid)

            loss = -torch.mean(v_estimate + IS)
        
        if self.loss_type!='Cloning':
            loss -= self.entropy_factor * self.entropy(context, estimated_value)
        
        return loss

    def entropy(self, context, estimated_value):
        x = torch.cat([context, estimated_value.reshape(-1,1)], dim=1)
        sigma = self.sigma(x)
        return (1. + torch.log(2 * np.pi * sigma.squeeze())).mean()/2
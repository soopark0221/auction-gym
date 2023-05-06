from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import device
import torch.nn as nn
from numba import jit
from torch.nn import functional as F
from tqdm import tqdm
import time


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
        nn.init.uniform_(self.bias_mu, -np.sqrt(3/self.weight_mu.size(1)), np.sqrt(3/self.weight_mu.size(1)))
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
    def __init__(self, context_dim, num_items, mode, rng, lr, c=2., nu=1.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = rng
        self.mode = mode
        self.lr = lr
        self.c = c
        self.nu = nu

        self.d = context_dim
        self.K = num_items
        self.m = torch.nn.Parameter(torch.Tensor(self.K, self.d))
        nn.init.kaiming_uniform_(self.m)

        self.S_inv = np.ones((self.K, self.d))
        self.S = torch.ones((self.K, self.d)).to(self.device)
        self.S0_inv = torch.ones((self.d,)).to(self.device)

        self.BCE = torch.nn.BCELoss(reduction='sum')
        self.eval()

        self.uncertainty = []
    
    def forward(self, X, A):
        return torch.sigmoid(torch.sum(X * self.m[A], dim=1))
    
    def update(self, contexts, items, outcomes, name):
        X = torch.Tensor(contexts).to(self.device)
        A = torch.LongTensor(items).to(self.device)
        y = torch.Tensor(outcomes).to(self.device)

        epochs = 100
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, amsgrad=True)

        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.loss(X, A, y, self.S0_inv)
            loss.backward()
            optimizer.step()
    
        y = self(X, A).numpy(force=True)
        X = contexts.reshape(-1,self.d)
        for k in range(self.K):
            mask = items==k
            y_ = y[mask]
            X_ = X[mask,:]
            N = y_.shape[0]
            y_ = y_ * (1 - y_)
            S_inv = self.S0_inv.numpy(force=True)
            S_inv += ((X_**2).T @ y_).reshape(-1)
            self.S_inv[k, :] = S_inv
            self.S[k,:] = torch.Tensor(np.sign(S_inv)/(np.abs(S_inv)+1e-2)).to(self.device)
            
    def loss(self, X, A, y, S0_inv):
        y_pred = self(X, A)
        loss = self.BCE(y_pred, y)
        for k in range(self.K):
            loss += torch.sum(self.m[k,:]**2 * S0_inv/2)
        return loss
    
    def estimate_CTR(self, context, UCB=False, TS=False):
        X = torch.Tensor(context.reshape(-1)).to(self.device)
        with torch.no_grad():
            if UCB:
                bound = self.c * torch.sqrt(self.S @ (X**2).T)
                self.uncertainty.append(bound.item())
                U = torch.sigmoid(torch.matmul(self.m, X) + bound)
                return U.numpy(force=True)
            elif TS:
                m = self.m.numpy(force=True)
                sqrt_S = np.sqrt(self.S.numpy(force=True))
                self.uncertainty.append(np.sqrt(np.sum(sqrt_S**2)))
                for k in range(self.K):
                    m[k,:] += self.nu * sqrt_S[k,:] * self.rng.normal(0,1,self.d)
                return (1 + np.exp(- m @ context))**(-1)
            else:
                return torch.sigmoid(torch.matmul(self.m, X)).numpy(force=True)
                # return torch.sigmoid(torch.matmul(self.m, X)).numpy(force=True)
    
    def get_uncertainty(self, index):
        if self.mode=='Epsilon-greedy':
            return np.array([0])
        else:
            return np.array(self.uncertainty[index:])

class LinearRegression:
    def __init__(self,context_dim, num_items, mode, rng, c=2.0, nu=1.0):
        super().__init__()
        self.rng = rng
        self.mode = mode # Epsilon-greedy or UCB or TS

        self.K = num_items
        self.d = context_dim

        self.N = np.zeros((self.K,), dtype=int)
        self.lambda0 = 1.0    # regularization constant
        self.c = c
        self.nu = nu

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
            m = self.m
            for k in range(self.K):
                m[k,:] += self.nu * self.sqrt_S[k,:,:] @ self.rng.normal(0,1,self.d)
            return m @ context
        else:
            return self.m @ context

    def get_uncertainty(self, index):
        eigvals = [np.linalg.eigvals(self.S[k,:,:]).reshape(-1) for k in range(self.K)]
        return np.concatenate(eigvals).real

class LogisticRegression(nn.Module):
    def __init__(self, context_dim, num_items, mode, rng, lr, c=1.0, nu=1.0):
        super().__init__()
        self.rng = rng
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.K = num_items
        self.d = context_dim
        self.c = c
        self.nu = nu

        self.m = nn.Parameter(torch.Tensor(self.K, self.d))
        nn.init.kaiming_uniform_(self.m)

        self.S0_inv = torch.Tensor(np.identity(self.d)).to(self.device)
        temp = [np.identity(self.d) for _ in range(self.K)]
        self.S_inv = np.stack(temp)
        self.sqrt_S = self.S_inv.copy()
        self.S = torch.Tensor(self.S_inv.copy()).to(self.device)

        self.BCE = torch.nn.BCELoss(reduction='sum')
        self.uncertainty = []
    
    def forward(self, X, A):
        return torch.sigmoid(torch.sum(X * self.m[A], dim=1))
    
    def update(self, contexts, items, outcomes, name):
        X = torch.Tensor(contexts).to(self.device)
        A = torch.LongTensor(items).to(self.device)
        y = torch.Tensor(outcomes).to(self.device)

        epochs = 100
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.loss(X, A, y, self.S0_inv)
            loss.backward()
            optimizer.step()
    
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
                x = y_[n] * X_[n,:].reshape(-1,1) @ X_[n,:].reshape(1,-1)
                S_inv += x
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
        with torch.no_grad():
            if UCB:
                bound = self.c * torch.sqrt(torch.tensordot(X,torch.tensordot(self.S, X, dims=([2],[0])), dims=([0],[1])))
                U = torch.sigmoid(torch.matmul(self.m, X) + bound)
                return U.numpy(force=True)
            elif TS:
                m = self.m.numpy(force=True)
                for k in range(self.K):
                    m[k,:] += self.nu * self.sqrt_S[k,:,:] @ self.rng.normal(0,1,self.d)
                return (1 + np.exp(- m @ context))**(-1)
            else:
                return torch.sigmoid(torch.matmul(self.m, X)).numpy(force=True)
    
    def get_uncertainty(self):
        S_ = self.S.numpy(force=True)
        eigvals = [np.linalg.eigvals(S_[k,:,:]).reshape(-1) for k in range(self.K)]
        return np.concatenate(eigvals).real

class LogisticRegressionM(nn.Module):
    def __init__(self, context_dim, items, mode, rng, lr, c=1.0, nu=1.0):
        super().__init__()
        self.rng = rng
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.items_np = items
        self.items = torch.Tensor(items).to(self.device)
        self.K = items.shape[0] # number of items
        self.d = context_dim
        self.h = items.shape[1] # item feature dimension
        self.c = c
        self.nu = nu

        self.M = nn.Parameter(torch.Tensor(self.d, self.h)) # CTR = sigmoid(context @ M @ item_feature)
        nn.init.kaiming_uniform_(self.M)

        self.BCE = torch.nn.BCELoss(reduction='sum')
        self.uncertainty = []
        self.S0_inv = torch.Tensor(np.eye(self.h*self.d)).to(self.device)
        self.S_inv = np.eye(self.h*self.d)
        self.S = torch.Tensor(np.eye(self.h*self.d)).to(self.device)
        self.sqrt_S = torch.Tensor(np.eye(self.h*self.d)).to(self.device)

    def forward(self, X, A):
        return torch.sigmoid(torch.sum(F.linear(X, self.M.T)*self.items[A], dim=1))
    
    def update(self, contexts, items, outcomes, name):
        X = torch.Tensor(contexts).to(self.device)
        A = torch.LongTensor(items).to(self.device)
        y = torch.Tensor(outcomes).to(self.device)

        epochs = 1000
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.loss(X, A, y)
            loss.backward()
            optimizer.step()
    
        y = self(X, A).numpy(force=True)
        y = y * (1 - y)
        contexts = contexts.reshape(-1,self.d)

        self.S_inv = self.S0_inv.numpy(force=True)
        for i in range(contexts.shape[0]):
            context = contexts[i]
            item_feature = self.items_np[A[i]]
            phi = np.outer(context, item_feature).reshape(-1)
            self.S_inv += y[i] * np.outer(phi, phi)
        self.S = torch.Tensor(np.diag(np.diag(self.S_inv)**(-1))).to(self.device)
        self.sqrt_S = torch.Tensor(np.diag(np.sqrt(np.diag(self.S_inv)+1e-6)**(-1))).to(self.device)

    def loss(self, X, A, y):
        y_pred = self(X, A)
        m = self.flatten(self.M)
        return self.BCE(y_pred, y) + torch.sum(m.T @ self.S0_inv @ m / 2)
    
    def estimate_CTR(self, context, UCB=False, TS=False):
        # context @ M @ item_feature = M * outer(context, item_feature)
        X = []
        context = context.reshape(-1)
        for i in range(self.K):
            X.append(np.outer(context, self.items_np[i]).reshape(-1))
        X = torch.Tensor(np.stack(X)).to(self.device)
        with torch.no_grad():
            if UCB:
                m = self.flatten(self.M)
                bound = self.c * torch.sum((X @ self.S) * X, dim=1, keepdim=True)
                mean = torch.sigmoid(X @ m).numpy(force=True).reshape(-1)
                U = torch.sigmoid(X @ m + bound).numpy(force=True).reshape(-1)
                return mean, U-mean
            elif TS:
                m = self.flatten(self.M)
                m = m + self.nu * self.sqrt_S @ torch.Tensor(self.rng.normal(0,1,self.d*self.h).reshape(-1,1)).to(self.device)
                out = torch.sigmoid(X @ m)
                return out.numpy(force=True).reshape(-1)
            else:
                m = self.flatten(self.M)
                return torch.sigmoid(X @ m).numpy(force=True).reshape(-1)

    def get_uncertainty(self):
        S_ = self.S.numpy(force=True)
        eigvals = np.linalg.eigvals(S_).reshape(-1)
        return eigvals.real

    def flatten(self, tensor):
        return torch.reshape(tensor, (tensor.shape[0]*tensor.shape[1], -1))
    
    def unflatten(self, tensor, x, y):
        return torch.reshape(tensor, (x, y))

class LogisticRegressionS(nn.Module):
    def __init__(self, context_dim, mode, rng, lr, c=1.0, nu=1.0):
        super().__init__()
        self.rng = rng
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.d = context_dim + 1
        self.c = c
        self.nu = nu

        self.m = nn.Parameter(torch.Tensor(1, self.d))
        nn.init.kaiming_uniform_(self.m)

        self.S0_inv = torch.Tensor(np.eye(self.d)).to(self.device)
        self.S_inv = np.eye(self.d)
        self.sqrt_S = self.S_inv.copy()
        self.S = torch.Tensor(self.S_inv.copy()).to(self.device)

        self.BCE = torch.nn.BCELoss(reduction='sum')
        self.uncertainty = []

    def forward(self, X):
        return torch.sigmoid(X @ self.m.T)

    def update(self, contexts, items, outcomes, name):
        X = torch.Tensor(contexts).to(self.device)
        y = torch.Tensor(outcomes).to(self.device)

        epochs = 100
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.loss(X, y, self.S0_inv)
            loss.backward()
            optimizer.step()
    
        y = self(X).numpy(force=True)
        N = y.shape[0]
        X = contexts.reshape(-1,self.d)
        y = y * (1 - y)
        S_inv = self.S0_inv.numpy(force=True)
        for n in range(N):
            x = y[n] * X[n].reshape(-1,1) @ X[n].reshape(1,-1)
            S_inv += x
        self.S_inv = S_inv

        S = np.diag(np.diag(S_inv)**(-1))
        self.sqrt_S = np.sqrt(S)
        self.S = torch.Tensor(S).to(self.device)
            
    def loss(self, X, y, S0_inv):
        y_pred = self(X).reshape(-1)
        loss = self.BCE(y_pred, y) + torch.sum(self.m @ S0_inv @ self.m.T/2)
        return loss
    
    def estimate_CTR(self, context, UCB=False, TS=False):
        X = torch.Tensor(context).to(self.device)
        with torch.no_grad():
            if UCB:
                bound = self.c * torch.sqrt(torch.sum((X @ self.S) * X, dim=1, keepdim=True))
                U = torch.sigmoid(X @ self.m.T + bound)
                return U.numpy(force=True).reshape(-1)
            elif TS:
                m = self.m.numpy(force=True)
                m += self.nu * self.sqrt_S @ self.rng.normal(0,1,self.d)
                return ((1 + np.exp(- context @ m.T))**(-1)).reshape(-1)
            else:
                return torch.sigmoid(X @ self.m.T).numpy(force=True).reshape(-1)
    
    def get_uncertainty(self):
        S_ = self.S.numpy(force=True)
        return np.diag(S_)

class BayesianNeuralRegression(nn.Module):
    def __init__(self, input_dim, latent_dim, prior_var):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = input_dim
        self.h = latent_dim
        self.prior_var = prior_var
        self.linear1 = BayesianLinear(self.d, self.h)
        self.linear2 = BayesianLinear(self.h, 1)
        self.criterion = nn.BCELoss()
        self.eval()

    def forward(self, x, MAP=False):
        x = torch.relu(self.linear1(x, not MAP))
        return torch.sigmoid(self.linear2(x, not MAP))

    def loss(self, predictions, labels, N):
        kl_div = self.linear1.KL_div(self.prior_var) + self.linear2.KL_div(self.prior_var)
        return self.criterion(predictions, labels) + kl_div/N
    
    def get_uncertainty(self):
        uncertainties = [self.linear1.get_uncertainty(),
                         self.linear2.get_uncertainty()]
        return np.concatenate(uncertainties)

class BayesianNeuralRegression2(nn.Module):
    def __init__(self, input_dim, latent_dim, prior_var):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = input_dim
        self.h = latent_dim
        self.prior_var = prior_var
        self.linear1 = BayesianLinear(self.d, self.h)
        self.linear2 = BayesianLinear(self.h, self.h)
        self.head = BayesianLinear(self.h, 1)
        self.criterion = nn.BCELoss()
        self.eval()

    def forward(self, x, MAP=False):
        x = torch.relu(self.linear1(x, not MAP))
        x = torch.relu(self.linear2(x, not MAP))
        return torch.sigmoid(self.head(x, not MAP))

    def loss(self, predictions, labels, N):
        kl_div = self.linear1.KL_div(self.prior_var) + self.linear2.KL_div(self.prior_var) + self.head.KL_div(self.prior_var)
        return self.criterion(predictions, labels) + kl_div/N
    
    def get_uncertainty(self):
        uncertainties = [self.linear1.get_uncertainty(),
                         self.linear2.get_uncertainty(),
                         self.head.get_uncertainty()]
        return np.concatenate(uncertainties)

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

class NeuralRegression2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = input_dim
        self.h = latent_dim
        self.linear1 = nn.Linear(self.d, self.h)
        self.linear2 = nn.Linear(self.h, self.h)
        self.head = nn.Linear(self.h, 1)
        self.BCE = nn.BCELoss()
        self.eval()

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
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
        self.linear1 = nn.Linear(self.d, self.h)
        self.linear2 = nn.Linear(self.h, self.h)
        self.heads = [nn.Linear(self.h, 1).to(self.device) for _ in range(num_heads)]
        self.BCE = nn.BCELoss()
        self.eval()
    
    def reset(self):
        self.linear1 = nn.Linear(self.d, self.h)
        self.linear2 = nn.Linear(self.h, self.h)
        self.heads = [nn.Linear(self.h, 1).to(self.device) for _ in range(self.num_heads)]
        self.to(self.device)
    
    def forward(self, x, i):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return torch.sigmoid(self.heads[i](x))

    def UCB_inference(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        y = [torch.sigmoid(self.heads[i](x)).numpy(force=True).reshape(-1) for i in range(self.num_heads)]
        y = np.stack(y)
        return np.mean(y, axis=0), np.std(y, axis=0)

    def TS_inference(self,x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        y = [torch.sigmoid(self.heads[i](x)).numpy(force=True).reshape(-1) for i in range(self.num_heads)]
        return np.stack(y)

    def loss(self, x, y, i):
        return self.BCE(self(x,i).squeeze(), y)
        

class LogisticWinRateEstimator(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(context_dim+1, 1),
            nn.Sigmoid()
        )
        self.BCE = nn.BCELoss()
        self.eval()

    def forward(self, x, sample=False):
        return self.ffn(x)
    
    def loss(self, x, y):
        return self.BCE(self.ffn(x), y)
    
    
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
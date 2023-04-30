import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.special import gamma, digamma

from Models import *


class Allocator:
    """ Base class for an allocator """

    def __init__(self, rng, item_features):
        self.rng = rng
        self.item_features = item_features
        self.feature_dim = item_features.shape[1]
        self.K = item_features.shape[0]

    def update(self, contexts, items, outcomes, name):
        pass

class AllocationDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
class NeuralAllocator(Allocator):
    def __init__(self, rng, item_features, lr, num_layers, num_epochs, context_dim, num_items, mode, latent_dim, eps=None, prior_var=None):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.num_epochs = num_epochs
        if self.mode=='Epsilon-greedy':
            if num_layers==1:
                self.net = NeuralRegression(context_dim+self.feature_dim, latent_dim).to(self.device)
                self.eps = eps
            else:
                self.net = NeuralRegression2(context_dim+self.feature_dim, latent_dim).to(self.device)
                self.eps = eps
        else:
            self.net = BayesianNeuralRegression(context_dim+self.item_features.shape[1], latent_dim, prior_var).to(self.device)
        self.count = 0
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    def update(self, contexts, items, outcomes, name):
        self.count += 1

        X = np.concatenate([contexts, self.item_features[items]], axis=1)
        X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes).to(self.device)
        N = X.shape[0]
        if N<10:
            return

        self.net.train()
        batch_size = 512

        for epoch in range(int(self.num_epochs)):
            shuffled_ind = self.rng.choice(N, size=N, replace=False)
            for i in range(int(N/batch_size)):
                self.optimizer.zero_grad()
                ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
                X_ = X[ind]
                y_ = y[ind]
                if self.mode=='Epsilon-greedy':
                    loss = self.net.loss(self.net(X_).squeeze(), y_)
                elif self.mode=='TS':
                    loss = self.net.loss(self.net(X_).squeeze(), y_, N)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
        self.net.eval()

    def estimate_CTR(self, context, TS=False):
        if TS:
            X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
            return self.net(X, MAP=False).numpy(force=True).reshape(-1)
        else:
            X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
            if self.mode=='TS':
                return self.net(X, MAP=True).numpy(force=True).reshape(-1)
            else:
                return self.net(X).numpy(force=True).reshape(-1)
    
    def get_uncertainty(self):
        if self.mode=='TS':
            return self.net.get_uncertainty()
        else:
            return np.array([0])

class LinearAllocator(Allocator):
    def __init__(self, rng, context_dim, mode, c=0.0, eps=0.1):
        super().__init__(rng)
        self.mode = mode # Epsilon-greedy or UCB or TS    
        self.d = context_dim
        self.c = c
        self.eps = eps

        self.model = LinearRegression(self.d, self.K, self.mode, rng, self.c)
        self.model.m = self.rng.normal(0 , 1/np.sqrt(self.d), (self.K, self.d))

    def update(self, contexts, items, outcomes, name):
        self.model.update(contexts, items, outcomes)

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)

    def get_uncertainty(self):
        return self.model.get_uncertainty()
    
class DiagLogisticAllocator(Allocator):
    def __init__(self, rng, lr, context_dim, num_items, mode, c=0.0, eps=0.1, nu=0.1):
        super().__init__(rng)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lr = lr

        self.c = c
        self.eps = eps
        self.nu = nu
        if self.mode=='UCB':
            self.model = DiagLogisticRegression(
                context_dim, num_items, self.mode, self.rng, self.lr, c=self.c).to(self.device)
        elif self.mode=='TS':
            self.model = DiagLogisticRegression(
                context_dim, num_items, self.mode, self.rng, self.lr, nu=self.nu).to(self.device)
        else:
            self.model = DiagLogisticRegression(
                context_dim, num_items, self.mode, self.rng, self.lr).to(self.device)

    def update(self, contexts, items, outcomes, name):
        self.model.update(contexts, items, outcomes, name)

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)
    
    def get_uncertainty(self):
        return self.model.get_uncertainty()
    
class LogisticAllocator(Allocator):
    def __init__(self, rng, lr, context_dim, num_items, mode, c=0.0, eps=0.1, nu=0.0):
        super().__init__(rng)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lr = lr

        self.K = num_items
        self.d = context_dim
        self.c = c
        self.eps = eps
        self.nu = nu
        if self.mode=='UCB':
            self.model = LogisticRegression(self.d, self.K, self.mode, self.rng, self.lr, c=self.c).to(self.device)
        elif self.mode=='TS':
            self.model = LogisticRegression(self.d, self.K, self.mode, self.rng, self.lr, nu=self.nu).to(self.device)
        else:
            self.model = LogisticRegression(self.d, self.K, self.mode, self.rng, self.lr).to(self.device)
        temp = [np.identity(self.d) for _ in range(self.K)]
        self.S0_inv = torch.Tensor(np.identity(self.d)).to(self.device)
        self.S_inv = np.stack(temp)
        self.S = torch.Tensor(self.S_inv.copy()).to(self.device)

        # self.initialize()
    
    def initialize(self):
        X = []
        for i in range(1000):
            context = self.rng.normal(0.0, 1.0, size=self.d)
            X.append(context/np.sqrt(np.sum(context**2)))
        X = np.stack(X)
        y = np.ones((1000,))
        A = self.rng.choice(self.K, (1000,))

        self.update(X, A, y, "")

    def update(self, contexts, items, outcomes, name):
        self.model.update(contexts, items, outcomes, name)

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)

    def get_uncertainty(self):
        return self.model.get_uncertainty()

class LogisticAllocatorM(Allocator):
    def __init__(self, rng, item_features, lr, context_dim, num_items, mode, c=0.0, eps=0.1, nu=0.0):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lr = lr

        self.K = num_items
        self.d = context_dim
        self.c = c
        self.eps = eps
        self.nu = nu
        if self.mode=='UCB':
            self.model = LogisticRegressionM(self.d, self.item_features, self.mode, self.rng, self.lr, c=self.c).to(self.device)
        elif self.mode=='TS':
            self.model = LogisticRegressionM(self.d, self.item_features, self.mode, self.rng, self.lr, nu=self.nu).to(self.device)
        else:
            self.model = LogisticRegressionM(self.d, self.item_features, self.mode, self.rng, self.lr).to(self.device)
        # self.initialize()
    
    def initialize(self):
        X = []
        for i in range(1000):
            context = self.rng.normal(0.0, 1.0, size=self.d)
            X.append(context/np.sqrt(np.sum(context**2)))
        X = np.stack(X)
        y = np.ones((1000,))
        A = self.rng.choice(self.K, (1000,))

        self.update(X, A, y, "")

    def update(self, contexts, items, outcomes, name):
        self.model.update(contexts, items, outcomes, name)

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)

    def get_uncertainty(self):
        return self.model.get_uncertainty()



class NeuralLogisticAllocator(Allocator):
    def __init__(self, rng, item_features, lr, context_dim, num_items, mode, latent_dim, c=1.0, eps=0.1, nu=0.1):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.mode = mode
        self.c = c
        self.eps = eps
        self.nu = nu

        self.d = context_dim
        self.K = num_items
        self.H = latent_dim
        self.net = NeuralRegression(self.d+self.feature_dim, self.H).to(self.device)

        if mode=='UCB':
            self.model = LogisticRegressionS(self.H, self.mode, self.rng, self.lr, c=self.c).to(self.device)
        elif mode=='TS':
            self.model = LogisticRegressionS(self.H, self.mode, self.rng, self.lr, nu=self.nu).to(self.device)
        else:
            self.model = LogisticRegressionS(self.H, self.mode, self.rng, self.lr).to(self.device)
        self.count = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
    def update(self, contexts, items, outcomes, name):
        self.count += 1

        X = np.concatenate([contexts, self.item_features[items]], axis=1)
        X = torch.Tensor(X).to(self.device)
        if self.count%5==0:
            y = torch.Tensor(outcomes).to(self.device)

            self.net.train()
            epochs = 1000

            for epoch in range(int(epochs)):
                self.optimizer.zero_grad()
                loss = self.net.loss(self.net(X).squeeze(), y)
                loss.backward()
                self.optimizer.step()
            self.net.eval()
        
        F = self.net.feature(X).numpy(force=True)
        # F = self.net.linear2(X)
        F = np.concatenate([F, np.ones((F.shape[0],1))],axis=1)
        self.model.update(F, items, outcomes, name)

    def estimate_CTR(self, context, UCB=False, TS=False):
        X = np.concatenate([np.tile(context.reshape(1,-1),(self.K,1)), self.item_features], axis=1)
        X = torch.Tensor(X).to(self.device)
        F = self.net.feature(X).numpy(force=True)
        # F = self.net.linear2(X)
        F = np.concatenate([F, np.ones((F.shape[0],1))],axis=1)
        return self.model.estimate_CTR(F, UCB, TS)

    def get_uncertainty(self):
        return self.model.get_uncertainty()

class NTKAllocator(Allocator):
    def __init__(self, rng, lr, context_dim, num_items, mode, c=1.0, nu=1.0):
        super().__init__(rng)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode    # TS or UCB
        self.lr = lr
        self.d = context_dim
        self.K = num_items
        self.nets = [NeuralRegression(self.d, 1).to(self.device) for _ in range(self.K)]
        for net in self.nets:
            net.initialize_weights()
        self.p = self.flatten(self.nets[0].parameters()).size(0)
        self.Z_inv = torch.Tensor(np.eye(self.p)).to(self.device)
        self.Z = torch.Tensor(np.eye(self.p)).to(self.device)
        self.uncertainty = np.zeros((self.K,))

        self.c = c
        self.nu = 0.1
        self.count = 0

    def update(self, contexts, items, outcomes, name):
        self.count += 1

        if self.count%10==0:
            for k in range(self.K):
                mask = items==k
                X, y = torch.Tensor(contexts[mask]).to(self.device), torch.Tensor(outcomes[mask]).to(self.device)
                N = X.size(0)
                self.nets[k].train()
                epochs = 100
                lr = 1e-3
                optimizer = torch.optim.Adam(self.nets[k].parameters(), lr=lr)
                for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
                    optimizer.zero_grad()
                    loss = self.nets[k].loss(self.nets[k].predict_item(X).squeeze(1), y)
                    loss.backward()
                    optimizer.step()
                self.nets[k].eval()

    def estimate_CTR(self, context, UCB=False, TS=False):
        X = torch.Tensor(context.reshape(-1)).to(self.device)
        g = self.grad(X)
        if UCB:
            bounds = []
            rets = []
            for k in range(self.K):
                mean = self.nets[k](X).numpy(force=True).squeeze()
                bound = self.c * torch.sqrt(((1/self.nets[k].H)*torch.matmul(torch.matmul(g[k],self.Z_inv),g[k]))).numpy(force=True)
                bounds.append(bound)
                ret = mean + bound
                rets.append(ret)
            bounds = np.stack(bounds)
            self.uncertainty = bounds.reshape(-1)
        elif TS:
            sigmas = []
            rets = []
            for k in range(self.K):
                mean = self.nets[k](X).numpy(force=True).squeeze()
                sigma = torch.sqrt(((1/self.nets[k].H)*torch.matmul(torch.matmul(g[k],self.Z_inv),g[k]))).numpy(force=True)
                ret =  mean + self.nu * sigma * self.rng.normal(0,1)
                rets.append(ret)
                sigmas.append(sigma)

            self.uncertainty = np.stack(sigmas).reshape(-1)
        max_k = np.argmax(rets)
        self.Z += torch.outer(g[max_k],g[max_k])/self.nets[k].H
        self.Z_inv = torch.inverse(torch.diag(torch.diag(self.Z))) 
        return rets

    def grad(self, X):
        g = []
        for k in range(self.K):
            y = self.nets[k](X).squeeze()
            g_k = torch.autograd.grad(outputs=y, inputs=self.nets[k].parameters())
            g_k = self.flatten(g_k)
            g.append(g_k)
        return torch.stack(g).to(self.device)

    def flatten(self, tensor):
        T=torch.tensor([]).to(self.device)
        for element in tensor:
            T=torch.cat([T,element.to(self.device).flatten()])
        return T

    def get_uncertainty(self):
        return self.uncertainty.flatten()



class OracleAllocator(Allocator):
    """ An allocator that acts based on the true P(click)"""

    def __init__(self, rng, item_features):
        super(OracleAllocator, self).__init__(rng, item_features)

    def set_CTR_model(self, M):
        self.M = M

    def estimate_CTR(self, context):
        return sigmoid(self.item_features @ self.M.T @ context / np.sqrt(context.shape[0]*self.item_features.shape[1]))
    
    def get_uncertainty(self):
        return np.array([0])

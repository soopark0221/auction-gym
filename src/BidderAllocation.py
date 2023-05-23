import numpy as np
import torch
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
    
class NeuralAllocator(Allocator):
    def __init__(self, rng, item_features, lr, batch_size, weight_decay, latent_dim, num_epochs, context_dim, num_items, eps):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.num_items = num_items
        self.context_dim = context_dim
        self.net = NeuralRegression(context_dim+self.feature_dim, latent_dim).to(self.device)
        self.mode = "Epsilon-greedy"
        self.eps = eps
        self.count = 0
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def update(self, contexts, items, outcomes, name):
        self.count += 1

        X = np.concatenate([contexts, self.item_features[items]], axis=1)
        X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes).to(self.device)
        N = X.shape[0]
        if N<10:
            return

        self.net.train()
        batch_size = min(N, self.batch_size)

        for epoch in range(int(self.num_epochs)):
            shuffled_ind = self.rng.choice(N, size=N, replace=False)
            for i in range(int(N/batch_size)):
                self.optimizer.zero_grad()
                ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
                X_ = X[ind]
                y_ = y[ind]
                loss = self.net.loss(self.net(X_).squeeze(), y_)
                loss.backward()
                self.optimizer.step()
        self.net.eval()
    
    def update_(self, contexts, features, outcomes):
        self.count += 1

        X = np.concatenate([contexts, features], axis=1)
        X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes).to(self.device)
        N = X.shape[0]
        if N<10:
            return

        self.net.train()
        batch_size = N
        for epoch in range(int(self.num_epochs)):
            shuffled_ind = self.rng.choice(N, size=N, replace=False)
            for i in range(int(N/batch_size)):
                self.optimizer.zero_grad()
                ind = shuffled_ind[i*batch_size:(i+1)*batch_size]
                X_ = X[ind]
                y_ = y[ind]
                loss = self.net.loss(self.net(X_).squeeze(), y_)
                loss.backward()
                self.optimizer.step()
        self.net.eval()

    def estimate_CTR(self, context, TS=False):
        X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
        return self.net(X).numpy(force=True).reshape(-1)
    
    def copy_param(self, ref):
        self.net.feature.weight.data = ref.net.feature.weight.clone().detach()
        self.net.feature.bias.data = ref.net.feature.bias.clone().detach()
        self.net.head.weight.data = ref.net.head.weight.clone().detach()
        self.net.head.bias.data = ref.net.head.bias.clone().detach()


class NeuralBootstrapAllocator(Allocator):
    def __init__(self, rng, item_features, lr, batch_size, weight_decay, latent_dim, num_epochs, context_dim,
                 num_items, num_heads, bootstrap, c):
        super().__init__(rng, item_features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.num_items = num_items
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.bootstrap = bootstrap
        self.c = c
        self.mode = "UCB"
        
        self.count = 0
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.net = MultiheadNeuralRegression(context_dim+self.feature_dim, latent_dim, num_heads).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def update(self, contexts, items, outcomes, name):
        self.count += 1

        X = np.concatenate([contexts, self.item_features[items]], axis=1)
        X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes).to(self.device)
        N = X.shape[0]
        if N<10:
            return

        self.net.train()
        batch_size = min(N, self.batch_size)
        ind_list = [np.random.choice(N, size=int(N*self.bootstrap)) for _ in range(self.num_heads)]

        for epoch in range(int(self.num_epochs)):
            for j in range(int(N*self.bootstrap/batch_size)):
                self.optimizer.zero_grad()
                loss = torch.tensor(0.0).to(self.device)
                for i in range(self.num_heads):
                    ind = ind_list[i][batch_size*j:batch_size*(j+1)]
                    X_ = X[ind]
                    y_ = y[ind]
                    loss += self.net.loss(X_, y_, i)
                loss.backward()
                self.optimizer.step()
        self.net.eval()
    
    def update_(self, contexts, features, outcomes):
        X = np.concatenate([contexts, features], axis=1)
        X, y = torch.Tensor(X).to(self.device), torch.Tensor(outcomes).to(self.device)
        N = X.shape[0]
        if N<10:
            return
        self.net.train()
        batch_size = min(N, self.batch_size)

        for epoch in range(int(self.num_epochs)):
            ind_list = [np.random.choice(N, size=int(N*self.bootstrap)) for _ in range(self.num_heads)]
            for i in range(self.num_heads):
                for j in range(int(N*self.bootstrap/batch_size)):
                    self.optimizer.zero_grad()
                    ind = ind_list[i][batch_size*j:batch_size*(j+1)]
                    X_ = X[ind]
                    y_ = y[ind]
                    loss = self.net.loss(X_, y_, i)
                    loss.backward()
                    self.optimizer.step()
        self.net.eval()

    def estimate_CTR(self, context, UCB=False):
        with torch.no_grad():
            if UCB:
                X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
                mean, std = self.net.UCB_inference(X)
                return mean, std
            else:
                X = torch.Tensor(np.concatenate([np.tile(context.reshape(1,-1),(self.K, 1)), self.item_features],axis=1)).to(self.device)
                mean, std = self.net.UCB_inference(X)
                return mean
    
    def copy_param(self, ref):
        self.net.linear.weight.data = ref.net.linear.weight.clone().detach()
        self.net.linear.bias.data = ref.net.linear.bias.clone().detach()
        for i in range(self.num_heads):
            self.net.heads[i].weight.data = ref.net.heads[i].weight.clone().detach()
            self.net.heads[i].bias.data = ref.net.heads[i].bias.clone().detach()


class OracleAllocator(Allocator):
    """ The optimal allocator """

    def __init__(self, rng, item_features):
        super(OracleAllocator, self).__init__(rng, item_features)

    def set_CTR_model(self, M):
        self.M = M

    def estimate_CTR(self, context):
        return sigmoid(self.item_features @ self.M.T @ context / np.sqrt(context.shape[0]))
    
    def get_uncertainty(self):
        return np.array([0])

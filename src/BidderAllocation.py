import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.special import gamma, digamma

from Models import *


class Allocator:
    """ Base class for an allocator """

    def __init__(self, rng):
        self.rng = rng

    def update(self, contexts, items, outcomes, name):
        pass
    
class NeuralAllocator(Allocator):
    def __init__(self, rng, lr, context_dim, num_items, mode, eps=0.1, prior_var=1.0):
        self.mode = mode # epsilon-greedy, BBB
        if self.mode=='Epsilon-greedy':
            self.net = NeuralRegression(n_dim=context_dim, n_items=num_items)
            self.eps = eps
        elif self.mode=='Bayes by Backprop':
            self.net = BayesianNeuralRegression(n_dim=context_dim, n_items=num_items, prior_var=prior_var)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.count = 0
        self.lr = lr
        super().__init__(rng)

    def update(self, contexts, items, outcomes, name):
        self.count += 1

        if self.count%10==0:
            X, A, y = contexts, items, outcomes
            N = X.shape[0]
            if N<10:
                return

            # self.net.train()
            # epochs = 100
            # optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

            # B = min(8192, N)
            # batch_num = int(N/B)

            # X, A, y = torch.Tensor(X).to(self.device), torch.LongTensor(A).to(self.device), torch.Tensor(y).to(self.device)
            # losses = []
            # for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            #     for i in range(batch_num):
            #         X_mini = X[i:i+B]
            #         A_mini = A[i:i+B]
            #         y_mini = y[i:i+B]
            #         optimizer.zero_grad()
            #         if self.mode=='Epsilon-greedy':
            #             loss = self.net.loss(self.net.predict_item(X_mini, A_mini).squeeze(), y_mini)
            #         elif self.mode=='Bayes by Backprop':
            #             loss = self.net.loss(self.net.predict_item(X_mini, A_mini).squeeze(), y_mini, N)
            #         loss.backward()
            #         optimizer.step()
            #         losses.append(loss.item())
            
            self.net.train()
            epochs = 100
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

            X, A, y = torch.Tensor(X).to(self.device), torch.LongTensor(A).to(self.device), torch.Tensor(y).to(self.device)
            losses = []
            for epoch in range(int(epochs)):
                optimizer.zero_grad()
                if self.mode=='Epsilon-greedy':
                    loss = self.net.loss(self.net.predict_item(X, A).squeeze(), y)
                elif self.mode=='Bayes by Backprop':
                    loss = self.net.loss(self.net.predict_item(X, A).squeeze(), y, N)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        self.net.eval()

    def estimate_CTR(self, context, sample=True):
        if self.mode=='Epsilon-greedy':
            return self.net(torch.from_numpy(context.astype(np.float32)).to(self.device)).numpy(force=True)
        elif self.mode=='Bayes by Backprop':
            return self.net(torch.from_numpy(context.astype(np.float32)).to(self.device), sample=sample).numpy(force=True)
    
    def get_uncertainty(self):
        if self.mode=='Bayes by Backprop':
            return self.net.get_uncertainty()
        else:
            return np.array([0])
    
class LinearAllocator(Allocator):
    def __init__(self, rng, context_dim, num_items, mode, c=0.0, eps=0.1):
        super().__init__(rng)
        self.mode = mode # Epsilon-greedy or UCB or TS    
        self.K = num_items
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
    def __init__(self, rng, lr, context_dim, num_items, mode, c=0.0, eps=0.1):
        super().__init__(rng)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lr = lr

        self.c = c
        self.eps = eps
        self.model = DiagLogisticRegression(context_dim, num_items, self.c, self.rng, self.lr).to(self.device)

    def update(self, contexts, items, outcomes, name):
        self.model.update(contexts, items, outcomes, name)

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)
    
    def get_uncertainty(self):
        return self.model.get_uncertainty()
    
class LogisticAllocator(Allocator):
    def __init__(self, rng, lr, context_dim, num_items, mode, c=0.0, eps=0.1):
        super().__init__(rng)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lr = lr

        self.K = num_items
        self.d = context_dim
        self.c = c
        self.eps = eps
        self.model = LogisticRegression(self.d, self.K, self.c, self.rng, self.lr).to(self.device)

        temp = [np.identity(self.d) for _ in range(self.K)]
        self.S0_inv = torch.Tensor(np.identity(self.d)).to(self.device)
        self.S_inv = np.stack(temp)
        self.S = torch.Tensor(self.S_inv.copy()).to(self.device)

    def update(self, contexts, items, outcomes, name):
        self.model.update(contexts, items, outcomes, name)

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)

    def get_uncertainty(self):
        return self.model.get_uncertainty()

class NeuralLinearAllocator(Allocator):
    def __init__(self, rng, lr, context_dim, num_items, mode, c=1.0, eps=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.d = context_dim
        self.K = num_items
        self.net = NeuralRegression(self.d, self.K, self.lr).to(self.device)

        split = mode.split()
        if split[0]=='Linear':
            self.model = LinearRegression(self.net.H, self.K, split[1], self.rng).to(self.device)
        elif split[0]=='Logistic':
            self.model = LogisticRegression(self.net.H, self.K, split[1], self.rng, self.lr).to(self.device)
        self.mode = split[1]

        self.c = c
        self.eps = eps
        self.count = 0
        super().__init__(rng)

    def update(self, contexts, items, outcomes, name):
        self.count += 1

        if self.count%10==0:
            X, A, y = torch.Tensor(contexts).to(self.device), torch.LongTensor(items).to(self.device), torch.Tensor(outcomes).to(self.device)
            N = y.size(0)

            self.net.train()
            epochs = 100
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

            losses = []
            for epoch in range(int(epochs)):
                optimizer.zero_grad()
                loss = self.net.loss(self.net.predict_item(X, A).squeeze(), y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            self.net.eval()
        
            F = self.net.feature(X).numpy(force=True)
            self.model.update(F, items, outcomes, name)

    def estimate_CTR(self, context, UCB=False, TS=False):
        X = torch.Tensor(context.reshape(-1)).to(self.device)
        F = self.net.feature(X).numpy(force=True)
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
        self.nets = [NeuralRegression(self.d, 1, self.lr).to(self.device) for _ in range(self.K)]

        #theta = self.flatten(self.nets[k].parameters())
        Z_size = len(self.flatten(self.nets[0].parameters()))
        Z_temp = [np.identity(Z_size) for _ in range(self.K)]
        #temp = [np.identity(self.d) for _ in range(self.K)]
        self.Z_inv = np.stack(Z_temp)
        self.Z = self.Z_inv.copy()
        self.uncertainty = np.zeros((self.K,))

        self.c = c
        self.nu = nu
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
                optimizer = torch.optim.Adam(self.nets[k].parameters(), lr=self.lr)

                for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
                        optimizer.zero_grad()
                        loss = self.nets[k].loss(self.nets[k].predict_item(X).squeeze(), y)
                        loss.backward()
                        optimizer.step()
                self.nets[k].eval()

    def estimate_CTR(self, context, UCB=False, TS=False):
        X = torch.Tensor(context.reshape(-1)).to(self.device)
        g = self.grad(X)
        if UCB:
            means = []
            bounds = []
            rets = []
            for k in range(self.K):
                mean = self.nets[k](X).numpy(force=True).squeeze()
                means.append(mean)
                bound = self.c * np.sqrt(np.tensordot(g[k], np.tensordot(np.expand_dims(self.Z_inv[k], axis=0), g[k], axes=([2],[0])), axes=([0],[1]))).squeeze()
                bounds.append(bound)
                ret = mean + bound
                rets.append(ret)
            means = np.stack(means)
            bounds = np.stack(bounds)
            self.uncertainty = bounds
        elif TS:
            means = []
            sigmas = []
            rets = []
            for k in range(self.K):
                mean = self.nets[k](X).numpy(force=True).squeeze()
                means.append(mean)
                sigma = self.nu * np.sqrt(np.tensordot(g[k], np.tensordot(np.expand_dims(self.Z_inv[k], axis=0), g[k], axes=([2],[0])), axes=([0],[1]))).squeeze()
                sigmas.append(sigma)
                ret =  mean + sigma * self.rng.normal(0,1)
                rets.append(ret)

            self.uncertainty = np.stack(sigmas)
        for k in range(self.K):
            self.Z[k] += np.outer(g[k],g[k])/self.nets[k].H
            self.Z_inv[k] -= self.Z_inv[k] @ (np.outer(g[k],g[k])/self.nets[k].H) @ self.Z_inv[k]/(1+ g[k]@self.Z_inv[k]@g[k])
        return ret

    def grad(self, X):
        g = []
        for k in range(self.K):
            y = self.nets[k](X).squeeze()
            #for param in self.nets[k].parameters():
            #    temp.append(param.grad.numpy(force=True))
            #g.append(np.concatenate(temp).flatten())
            g_k = torch.autograd.grad(outputs=y, inputs=self.nets[k].parameters())
            g_k = self.flatten(g_k)
            g.append(g_k)
        return np.stack(g)

    def flatten(self, tensor):
        T=torch.tensor([]).to('cpu')
        for element in tensor:
            T=torch.cat([T, element.to('cpu').flatten()])
        return T.detach().numpy()

    def get_uncertainty(self):
        return self.uncertainty.flatten()

class OracleAllocator(Allocator):
    """ An allocator that acts based on the true P(click)"""

    def __init__(self, rng):
        self.item_features = None
        self.activation = None
        super(OracleAllocator, self).__init__(rng)

    def update_item_features(self, item_features):
        self.item_features = item_features

    def estimate_CTR(self, context):
        if self.activation=='Linear':
            return 0.5 + 0.5 * context @ self.item_features.T
        else:    # Logistic
            return sigmoid(context @ self.item_features.T)
    
    def get_uncertainty(self):
        return np.array([0])

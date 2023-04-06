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


class DiagLogisticAllocator(Allocator):
    """ An allocator that estimates P(click) with Logistic Regression implemented in PyTorch"""

    def __init__(self, rng, context_dim, num_items, mode, c=0.0):
        self.response_model = DiagLogisticRegression(n_dim=context_dim, n_items=num_items)
        self.mode = mode
        self.c = c
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.response_model.to(self.device)
        super().__init__(rng)

    def update(self, contexts, items, outcomes, name):
        # Rename
        X, A, y = contexts, items, outcomes

        if len(y) < 2:
            return

        # Fit the model
        self.response_model.train()
        epochs = 2000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.response_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

        X, A, y = torch.Tensor(X).to(self.device), torch.LongTensor(A).to(self.device), torch.Tensor(y).to(self.device)
        losses = []
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()  # Setting our stored gradients equal to zero
            loss = self.response_model.loss(torch.squeeze(self.response_model.predict_item(X, A)), y)
            loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
            optimizer.step()  # Updates weights and biases with the optimizer (SGD)
            losses.append(loss.item())
            scheduler.step(loss)

            if epoch > 512 and np.abs(losses[-100] - losses[-1]) < 1e-6:
                print(f'Stopping at Epoch {epoch}')
                break

        # Laplace Approximation for variance q
        with torch.no_grad():
            for item in range(self.response_model.m.shape[0]):
                item_mask = items == item
                X_item = torch.Tensor(contexts[item_mask]).to(self.device)
                self.response_model.laplace_approx(X_item, item)
            self.response_model.update_prior()

        self.response_model.eval()

    def estimate_CTR(self, context, UCB=False, MAP=False):
        if UCB:
            return self.response_model(torch.Tensor(context).to(self.device)).numpy(force=True) + \
                self.c * np.sqrt(np.tensordot(context.T,np.tensordot(self.response_model.q.numpy(force=True), context, axes=([2],[0])), axes=([0],[1])))
        else:
            return self.response_model(torch.Tensor(context).to(self.device)).numpy(force=True)
    
    def get_uncertainty(self):
        return self.response_model.get_uncertainty()
    
class NeuralAllocator(Allocator):
    def __init__(self, rng, context_dim, num_items, mode, eps=0.1, prior_var=1.0):
        self.mode = mode # epsilon-greedy, BBB
        if self.mode=='Epsilon-greedy':
            self.net = NeuralRegression(n_dim=context_dim, n_items=num_items)
            self.eps = eps
        elif self.mode=='Bayes by Backprop':
            self.net = BayesianNeuralRegression(n_dim=context_dim, n_items=num_items, prior_var=prior_var)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.count = 0
        super().__init__(rng)

    def update(self, contexts, items, outcomes, name):
        self.count += 1

        if self.count%10==0:
            X, A, y = contexts, items, outcomes
            N = X.shape[0]
            if N<10:
                return

            self.net.train()
            epochs = 100
            lr = 1e-3
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

            B = min(8192, N)
            batch_num = int(N/B)

            X, A, y = torch.Tensor(X).to(self.device), torch.LongTensor(A).to(self.device), torch.Tensor(y).to(self.device)
            losses = []
            for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
                for i in range(batch_num):
                    X_mini = X[i:i+B]
                    A_mini = A[i:i+B]
                    y_mini = y[i:i+B]
                    optimizer.zero_grad()
                    if self.mode=='Epsilon-greedy':
                        loss = self.response_model.loss(self.response_model.predict_item(X_mini, A_mini).squeeze(), y_mini)
                    elif self.mode=='Bayes by Backprop':
                        loss = self.response_model.loss(self.response_model.predict_item(X_mini, A_mini).squeeze(), y_mini, N)
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

        self.model = LinearRegression(self.d, self.K, self.mode, self.c)
        self.model.m = self.rng.normal(0 , 1/np.sqrt(self.d), (self.K, self.d))

    def update(self, contexts, items, outcomes, name):
        self.model.update(contexts, items, outcomes)
        print(f"{name} : allocator updated")

    def estimate_CTR(self, context, UCB=False, TS=False):
        return self.model.estimate_CTR(context, UCB, TS)

    def get_uncertainty(self):
        return self.model.get_uncertainty()
    
class LogisticAllocator(Allocator):
    def __init__(self, rng, context_dim, num_items, mode, c=0.0, eps=0.1):
        super().__init__(rng)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode

        self.K = num_items
        self.d = context_dim
        self.c = c
        self.eps = eps
        self.model = LogisticRegression(self.d, self.K, self.c, self.rng).to(self.device)

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
    def __init__(self, rng, context_dim, num_items, mode, c=1.0, eps=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d = context_dim
        self.K = num_items
        self.H = 16    # hidden layer dimension
        self.net = NeuralRegression(self.d, self.K).to(self.device)

        split = mode.split()
        if split[0]=='Linear':
            self.model = LinearRegression(self.H, self.K, split[1], self.rng).to(self.device)
        elif split[0]=='Logistic':
            self.model = LogisticRegression(self.H, self.K, split[1], self.rng).to(self.device)
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
            lr = 1e-3
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

            B = min(8192, N)
            batch_num = int(N/B)
            losses = []
            for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
                for i in range(batch_num):
                    X_mini = X[i:i+B]
                    A_mini = A[i:i+B]
                    y_mini = y[i:i+B]
                    optimizer.zero_grad()
                    loss = self.net.loss(self.net.predict_item(X_mini, A_mini).squeeze(), y_mini)
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

class OracleAllocator(Allocator):
    """ An allocator that acts based on the true P(click)"""

    def __init__(self, rng):
        self.item_embeddings = None
        super(OracleAllocator, self).__init__(rng)

    def update_item_embeddings(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def estimate_CTR(self, context):
        return sigmoid(self.item_embeddings @ context / np.sqrt(context.shape[0]))
    
    def get_uncertainty(self):
        return np.array([0])

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Models import PyTorchLogisticRegression, NeuralRegression, sigmoid


class Allocator:
    """ Base class for an allocator """

    def __init__(self, rng):
        self.rng = rng

    def update(self, contexts, items, outcomes, name):
        pass


class PyTorchLogisticRegressionAllocator(Allocator):
    """ An allocator that estimates P(click) with Logistic Regression implemented in PyTorch"""

    def __init__(self, rng, embedding_size, num_items, thompson_sampling=True):
        self.response_model = PyTorchLogisticRegression(n_dim=embedding_size, n_items=num_items)
        self.thompson_sampling = thompson_sampling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.response_model.to(self.device)
        super(PyTorchLogisticRegressionAllocator, self).__init__(rng)

    def update(self, contexts, items, outcomes, iteration, plot, figsize, fontsize, name):
        # Rename
        X, A, y = contexts, items, outcomes

        if len(y) < 2:
            return

        # Fit the model
        self.response_model.train()
        epochs = 8192 * 2
        lr = 2e-3
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

            if epoch > 1024 and np.abs(losses[-100] - losses[-1]) < 1e-6:
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

    def estimate_CTR(self, context, sample=True):
        return self.response_model(torch.from_numpy(context.astype(np.float32)).to(self.device),
                                   sample=(self.thompson_sampling and sample)).numpy(force=True)
    
class NeuralAllocator(Allocator):
    def __init__(self, rng, context_dim, num_items, thompson_sampling=True):
        self.response_model = NeuralRegression(n_dim=context_dim, n_items=num_items)
        self.thompson_sampling = thompson_sampling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.response_model.to(self.device)
        super().__init__(rng)

    def update(self, contexts, items, outcomes, name):
        # Rename
        X, A, y = contexts, items, outcomes
        N = X.shape[0]

        if len(y) < 2:
            return

        # Fit the model
        self.response_model.train()
        epochs = 10000
        lr = 1e-3
        optimizer = torch.optim.Adam(self.response_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

        B = min(4096, N)
        batch_num = int(N/B)

        X, A, y = torch.Tensor(X).to(self.device), torch.LongTensor(A).to(self.device), torch.Tensor(y).to(self.device)
        losses = []
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            for i in range(batch_num):
                X_mini = X[i:i+B]
                A_mini = A[i:i+B]
                y_mini = y[i:i+B]
                optimizer.zero_grad()
                loss = self.response_model.loss(self.response_model.predict_item(X_mini, A_mini).squeeze(), y_mini, N)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                scheduler.step(loss)

            if epoch > 1024 and np.abs(losses[-100] - losses[-1]) < 1e-6:
                print(f'Stopping at Epoch {epoch}')
                break
        self.response_model.eval()

    def estimate_CTR(self, context, sample=True):
        return self.response_model(torch.from_numpy(context.astype(np.float32)).to(self.device),
                                   sample=(self.thompson_sampling and sample)).numpy(force=True)


class OracleAllocator(Allocator):
    """ An allocator that acts based on the true P(click)"""

    def __init__(self, rng):
        self.item_embeddings = None
        super(OracleAllocator, self).__init__(rng)

    def update_item_embeddings(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def estimate_CTR(self, context):
        return sigmoid(self.item_embeddings @ context)

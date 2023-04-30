import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numba import jit
from torch.nn import functional as F
from tqdm import tqdm
import argparse
import json
import pandas as pd
import seaborn as sns
import os

from BidderAllocation import NeuralAllocator

NUM_RUNS = 3
CONTEXT_DIM = 5
FEATURE_DIM = 5
NUM_ITEMS = 10
NUM_DATA = 50000
FIGSIZE = (8, 5)
FONTSIZE = 14

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def draw_feature():
    temp = []
    for k in range(NUM_ITEMS):
        feature = np.random.normal(0.0, 1.0, size=FEATURE_DIM)
        temp.append(feature)
    M = np.random.normal(0.0, 1.0, size=(CONTEXT_DIM, FEATURE_DIM))
    return np.stack(temp), M

def generate_context():
    return np.random.normal(0.0, 1.0, size=CONTEXT_DIM)

def CTR(context, M, item_features):
    # CTR = sigmoid(context @ M @ item_feature) for each item
    return sigmoid(item_features @ M.T @ context / np.sqrt(CONTEXT_DIM*FEATURE_DIM))

# def validate(model : NeuralAllocator):
#     prob = []
#     pred = []
#     for _ in range(500):
#         c = generate_context()
#         ctr = CTR(c, M, feature).reshape(-1)
#         prob.append(ctr)
#         pred.append(model.estimate_CTR(c, False).reshape(-1))
#     prob = np.concatenate(prob)
#     pred = np.concatenate(pred)
#     return np.sqrt(np.sum((pred-prob)**2)/len(prob))

def validate(model : NeuralAllocator, features):
    X = []
    prob = []
    for _ in range(500):
        c = generate_context()
        x = np.concatenate([np.tile(c.reshape(1,-1),(NUM_ITEMS, 1)), features],axis=1)
        X.append(x)
        ctr = CTR(c, M, feature).reshape(-1)
        prob.append(ctr)
    X = torch.Tensor(np.concatenate(X)).to(model.device)
    pred = model.net(X).numpy(force=True).reshape(-1)
    prob = np.concatenate(prob)
    return np.sqrt(np.sum((pred-prob)**2)/len(prob))

if __name__=='__main__':
    np.random.seed(42)
    rng = np.random.default_rng(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    with open(args.config) as f:
        config = json.load(f)
    
    config = config['agents'][1]['allocator']['kwargs']
    
    error = []

    for run in range(NUM_RUNS):
        feature, M = draw_feature()
        model = NeuralAllocator(rng, feature, config['lr'], config['batch_size'], config['weight_decay'], config['num_layers'], config['latent_dim'], config['num_epochs'],
                                CONTEXT_DIM, NUM_ITEMS, 'Epsilon-greedy', config['eps'])

        context = []
        item = []
        outcome = []
        for _ in range(NUM_DATA):
            c = generate_context()
            i = np.random.choice(NUM_ITEMS)
            context.append(c)
            item.append(i)
            ctr = CTR(c, M, feature[i]).item()
            outcome.append(np.random.binomial(1,ctr))
        context = np.stack(context)
        item = np.stack(item)
        outcome = np.stack(outcome)

        temp = []
        for i in tqdm(range(100)):
            model.update(context, item, outcome, '...')
            if (i+1)%5==0:
                val = validate(model, feature)
                print('valudation: ' + str(val))
                temp.append(val)
        error.append(temp)
        
    
    name = f"lr_{config['lr']}_layers_{config['num_layers']}_latent_{config['latent_dim']}_epoch_{config['num_epochs']}"

    df_rows = {'Run': [], 'Step': [], 'Error': []}
    for run, list in enumerate(error):
        for step, number in enumerate(list):
                df_rows['Run'].append(run)
                df_rows['Step'].append(step)
                df_rows['Error'].append(number)
    df = pd.DataFrame(df_rows)

    fig, axes = plt.subplots(figsize=FIGSIZE)
    plt.title('CTR prediction error', fontsize=FONTSIZE + 2)
    sns.lineplot(data=df, x="Step", y="Error", ax=axes)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.ylabel('Error', fontsize=FONTSIZE)
    plt.xlabel(f"x{5} epochs", fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.tight_layout()
    plt.savefig(f"results/NN_tuning/{name}.png", bbox_inches='tight')
## Efficient Multi-Agent Auction Bidding in Online Ad-Auction  

This repository contains the source code for Multi-agent Auction Bidding. It is based on AuctionGym simulation environment [paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym) that enables reproducible offline evaluation of bandit and reinforcement learning approaches to ad allocation and bidding in online advertising auctions.

## Getting Started 
This section provides instructions to reproduce the results.

We provide a script that takes as input a configuration file detailing the environment and bidders (in JSON format), and outputs raw logged metrics over repeated auction rounds in .csv-files, along with visualisations.
To reproduce the results for UCB, run:

```
python src/main.py config/FP_DQN_Logi-UCB-MB.json
```

A `results`-directory will be created, with a subdirectory per configuration file that was ran. This subdirectory will contain .csv-files with raw metrics, and .pdf-files with general visualisations.
Other configuration files will generate results for greedy, dithering, and Thompson sampliong method.

## Efficient Multi-Agent Auction Bidding in Online Ad-Auction  

This is the source code for Multi-agent Auction Bidding. It is based on AuctionGym simulation environment. [paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym)

## Requirements
PyTorch, version: 1.13.1

```
pip install -r requirements.txt
```

## How to Run
We provide a script that takes as input a configuration file detailing the environment and bidders (in JSON format), and outputs raw logged metrics over repeated auction rounds in .csv-files, along with visualisations.
To reproduce the results for UCB, run:

```
python src/main.py config/FP_DQN_Logi-UCB.json
```

A `results`-directory will be created, with a subdirectory per configuration file that was ran. This subdirectory will contain .csv-files with raw metrics, and .pdf-files with general visualisations.
Other configuration files will generate results for greedy, dithering, and Thompson sampling method.

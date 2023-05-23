### Efficient Exploration for Online Advertising Auctions

Source code for Double-UCB algorithm and baseline methods. The simulation environment is based on [AuctionGym](https://github.com/amzn/auction-gym.git). Networks are implemented using Pytorch 2.0 and Python 3.10.

## Reproducing Results

The config files are located on config/ directory. The files `training.json` and `training_multibid.json` contains the configurations of the auction environment and training. The other config files describes agents participating auctions. To reproduce the results of Double-UCB, run:
```
python src/main.py config/Bootstrap-UCB.json
```
The results of multi-bidding can be reproduced by:
```
python src/main_multibid.py [agent_config_path]
```
The results will be saved in the `results/` directory.

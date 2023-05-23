#### Training Config

| Key  | Description |
| ------------- | ------------- |
| `random_seed` | The random seed that is used as input to the random number generator  |
| `num_participants_per_round` | The number of participants in every auction round |
| `num_runs` | The number of runs to repeat and average results over  |
| `num_iter` | The time horizon |
| `record_interval` | The period of recording statistics |
| `update_interval` | The period of model update |
| `context_dim` | The dimension of context vectors |
| `feature_dim` | The dimension of ad feature vectors |
| `context_distribution` | The distribution of context vectors |
| `explore_then_commit` | The number of warm-up steps. Used only for baselines DM, IPS, and DR |
| `allocation` | The type of allocation. Second place auction is possible, yet not used.  |


#### Training Config (Multi-bid)

| Key  | Description |
| ------------- | ------------- |
| `random_seed` | The random seed that is used as input to the random number generator  |
| `num_participants_per_round` | The number of participants in every auction round |
| `num_runs` | The number of runs to repeat and average results over  |
| `num_iter` | The time horizon |
| `record_interval` | The period of recording statistics |
| `update_interval` | The period of model update |
| `context_dim` | The dimension of context vectors |
| `feature_dim` | The dimension of ad feature vectors |
| `context_distribution` | The distribution of context vectors |
| `allocation` | The type of allocation. Second place auction is possible, yet not used.  |
| `num_auctions` | The number of auctions |
| `bonus_factor` | The pool of c_alc (count-based allocation) |
| `optimism_scale` | The pool of c_opt (optimistic estimation) |
| `overbidding_factor` | The pool of c_over (count-based overbidding) |


#### Agent Config

| Key  | Description |
| ------------- | ------------- |
| `name` | An identifier for the agent  |
| `num_copies` | The number of agents with this configuration (but unique item catalogues). A suffix will be appended to the name if num_copies > 1 |
| `num_items` | The number of items in the ad catalogue |
| `allocator` | The config for allocator |
| `bidder` | The config for bidder |
| `bonus_factor` | The pool of c_alc (count-based allocation) |


Allocators have types, and possible keyword arguments supporting those types. Possible allocators are `OracleAllocator`, `NeuralAllocator`(Epsilon-Greedy and Greedy), and `NeuralBootstrapAllocator`(Bootstrap-UCB).

Bidders are `TruthfulBidder`(for competitors), `OracleBidder`, `DefaultBidder`, and `StochasticPolicyBidder`(for IPS and DR)
c_opt and c_over are specified by the `kwargs` field of the bidder config.

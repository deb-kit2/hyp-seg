# Hyperbolic Graph Neural Networks
This code has been adapted from [HazyResearch/hgcn](https://github.com/HazyResearch/hgcn/tree/master), [JindouHai/H2H-GCN](https://github.com/JindouDai/H2H-GCN/tree/main) and [SAMPL-Weizmann/DeepCut](https://github.com/SAMPL-Weizmann/DeepCut).

## To-Do :
 - [ ] fix cuda error.
 - [ ] maybe remove log_map_zero from model output.
 - [ ] Print logs.
 - [ ] OverRide function in `main.py`.
 - [ ] play with args.num_layers
 - [x] `args.step` overlapping.
 - [x] Fix arguments.
 - [x] load_data
 - [x] init model, forward_pass and loss
 - [x] check norm of F
 - [x] check feature_dim
 - [x] DeepCut
 - [x] Check degrees of freedom.
 - [x] test backprop with radam
 - [x] check hgcn issues.
 - [x] Graph layer.
 - [x] Init Code.

## Study about :
 - [x] Distance matrix normalization.
 - [x] Similarity as a dot product still valid?
        - Not exactly.
        - But, can go from distance, to exponential, to get similarity.
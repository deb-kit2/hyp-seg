# Hyperbolic Graph Neural Networks
This code has been adapted from [HazyResearch/hgcn](https://github.com/HazyResearch/hgcn/tree/master) and [JindouHai/H2H-GCN](https://github.com/JindouDai/H2H-GCN/tree/main).

## To-Do :
 - [ ] main.py
 - [ ] OverRide function in main.py
 - [ ] Transform graph output back to Euclidean
 - [x] DeepCut
 - [x] Check degrees of freedom.
 - [x] test backprop with radam
 - [x] check hgcn issues.
 - [x] Graph layer.
 - [x] Init Code.

## Study about :
 - [ ] Distance matrix normalization.
 - [x] Similarity as a dot product still valid?
        - Not exactly.
        - But, can go from distance, to exponential, to get similarity.
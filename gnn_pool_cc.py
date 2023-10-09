import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.h2hgcn import H2HGCN


class GNNpool(nn.Module) :
    # def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters) :
    def __init__(self, args, mlp_hidden, num_clusters) :
        """
        implementation of mincutpool model from: https://arxiv.org/pdf/1907.00481v6.pdf
        @param input_dim: Size of input nodes features
        @param conv_hidden: Size Of conv hidden layers
        @param mlp_hidden: Size of mlp hidden layers
        @param num_clusters: Number of cluster to output
        @param device: Device to run the model on
        """

        super(GNNpool, self).__init__()
        self.device = args.device
        self.num_clusters = num_clusters
        self.mlp_hidden = mlp_hidden

        # GNN conv
        self.convs = H2HGCN(args, last = True)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(args.dim, mlp_hidden), nn.ELU(), nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))
        
        self.args.eucl_vars.append(self.mlp)

    def forward(self, x, adj, edge_weights) :
        """
        forward pass of the model
        @param data: Graph in Pytorch geometric data format
        @param A: Adjacency matrix of the graph
        @return: Adjacency matrix of the graph and pooled graph (argmax of S)
        """
        x = self.convs.encode(x, adj, edge_weights)  # applying conv
        x = F.elu(x)

        # pass feats through mlp
        H = self.mlp(x)

        # cluster assignment for matrix S
        S = F.softmax(H)

        return adj, S

    def loss(self, A, S) :
        """
        loss calculation, relaxed form of Normalized-cut
        @param A: Adjacency matrix of the graph
        @param S: Polled graph (argmax of S)
        @return: loss value
        """
        # cc loss
        X = torch.matmul(S, S.t())
        cc_loss = -torch.sum(A * X)

        return cc_loss
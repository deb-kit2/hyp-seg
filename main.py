import os
import json
import argparse
import logging

from manifolds.stiefel import StiefelManifold
from manifolds.lorentz import LorentzManifold
from utils.pre_utils import set_seed


def parse_args() :
    parser = argparse.ArgumentParser()

    # over-ride if given
    parser.add_argument("--config", type = str)

    parser.add_argument("--exp_name", help = "Experiment name", type = str, required = True)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--cuda", type = int, default = 0)

    parser.add_argument("--epochs", type = int)
    parser.add_argument("--patience", type = int)

    # graph parameters
    parser.add_argument("--feature_dim", type = int, default = 256)
    parser.add_argument("--dim", type = int, default = 16)
    parser.add_argument("--num_layers", type = int, default = 1)
    parser.add_argument("--tie_weight", action = "store_true")
    parser.add_argument("--euclidean_lr", type = float)
    parser.add_argument("--stiefel_lr", type = float)
    
    # image parameters
    
    args = parser.parse_args()
    return args


def train() :
    
    return


if __name__ == "__main__" :
    args = parse_args()
    
    # logging
    if not os.path.isdir("logs") :
        os.mkdir("logs")

    logging.basicConfig(level = logging.DEBUG,
                        handlers = [
                            logging.FileHandler("logs/" + args.exp_name.split("/")[-1] + ".log"),
                            logging.StreamHandler()
                        ],
                        format = "%(levelname)s : %(message)s")
    
    if args.config is not None :
        logging.warning(f"Over-riding arguments from {args.config}.")
        raise NotImplementedError

    logging.info("\n" + json.dumps({key : value for key, value in vars(args).items()}, indent = 4) + "\n")

    args.manifold = LorentzManifold(args)
    args.select_manifold = "lorentz"
    args.dim = args.dim + 1
    args.weight_manifold = StiefelManifold(args, 1)
    args.proj_init = "orthogonal"

    args.stie_vars = []
    args.eucl_vars = []

    args.device = "cuda:" + str(args.cuda) if int(args.cuda) >= 0 else "cpu"
    args.patience = args.epochs if not args.patience else int(args.patience)
    set_seed(args.seed)

    train()
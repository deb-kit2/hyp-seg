import os
import json
import argparse
import logging

from utils.extractor import ViTExtractor
from utils.features_extract import deep_features
from utils.bilateral_solver import bilateral_solver_output

from manifolds.stiefel import StiefelManifold
from manifolds.lorentz import LorentzManifold

from tqdm import tqdm
from utils.pre_utils import set_seed
from utils.util import download_url, load_data_img, create_adj


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
    
    # segmentation parameters
    parser.add_argument("--mode", type = int, default = 0, choices = [0, 1, 2],
                        help = "0 for 1-stage segmentation\n1 for 2-stage segmentation on foreground\n2 for 2-stage segmentation on both background and foreground")
    parser.add_argument("--cut", type = str, default = "cc", choices = ["cc", "ncut"])
    parser.add_argument("--alpha", type = int, default = 3, help = "alpha for CC loss")
    parser.add_argument("--step", type = int, default = 1)
    parser.add_argument("--K", type = int, default = 2, help = "Number of clusters")
    parser.add_argument("--cc", action = "store_true", default = False,
                        help = "Show only largest component in segmentation map (for k == 2)")
    parser.add_argument("--bs", action = "store_true", default = False,
                        help = "Apply bilateral solver")
    parser.add_argument("--log_bin", action = "store_true", default = False,
                        help = "Apply log binning to extracted descriptors (correspond to smoother segmentation maps)")

    # image parameters
    parser.add_argument("--res", type = int, default = 280)
    parser.add_argument("--stride", type = int, default = 8)
    parser.add_argument("--facet", type = str, default = "key", choices = ["key", "query", "value"], 
                        help = "Facet fo descriptor extraction")
    parser.add_argument("--layer", type = int, default = 11)
    parser.add_argument("--pretrained_weights", type = str, 
                        default = './dino_deitsmall8_pretrain_full_checkpoint.pth')

    # path parameters
    parser.add_argument("--input_dir", type = str, required = True)
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--save", action = "store_true", default = False)

    args = parser.parse_args()
    return args


def train() :
    extractor = ViTExtractor('dino_vits8', args.stride, model_dir = args.pretrained_weights, device = args.device)
    
    feats_dim = 6528 if args.log_bin else 384
    if args.mode in [1, 2] :
        foreground_k = K
        K = 2
    
    # model init
    if args.cut == 0 : 
        from gnn_pool import GNNpool
    else : 
        from gnn_pool_cc import GNNpool

    for file in tqdm(os.listdir(args.input_dir)) :
        if not file.endswith((".jpg", ".png", ".jpeg")) :
            continue

        if os.path.exists(os.path.join(args.output_dir, file.split('.')[0] + '.txt')) :
            continue
        if os.path.exists(os.path.join(args.output_dir, file)) :
            continue

        image_tensor, image = load_data_img(os.path.join(args.input_dir, file), args.res)
        
        # Extract deep features, from the transformer and create an adj matrix
        F = deep_features(image_tensor, extractor, 
                          args.layer, args.facet, bin = args.log_bin, device = args.device)
        W = create_adj(F, args.cut, args.alpha)


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

    # Check input and output paths
    if not os.path.isdir(args.input_dir) :
        raise Exception("Image input dir does not exist.")
    if not os.path.isdir(args.output_dir) :
        os.mkdir(args.output_dir)
    if not os.path.exists(args.pretrained_weights) :
        logging.info("Pretrained weights not found for ViTs. Downloading...")
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        download_url(url, args.pretrained_weights)

    assert not(args.K != 2 and args.cc), 'largest connected component only available for k == 2'
    args.cut = 0 if args.cut == "ncut" else 1
    if args.cut == 1 :
        K = 10

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
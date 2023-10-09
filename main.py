import os
import json
import argparse
import logging

import torch

from utils.extractor import ViTExtractor
from utils.features_extract import deep_features
from utils.bilateral_solver import bilateral_solver_output

from manifolds.stiefel import StiefelManifold
from manifolds.lorentz import LorentzManifold

from tqdm import tqdm
from utils.pre_utils import set_seed, set_up_optimizer_scheduler
from utils.util import *


def parse_args() :
    parser = argparse.ArgumentParser()

    # over-ride if given
    parser.add_argument("--config", type = str)

    parser.add_argument("--exp_name", help = "Experiment name", type = str, required = True)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--cuda", type = int, default = 0)

    parser.add_argument("--epochs", type = int)
    parser.add_argument("--patience", type = int)

    parser.add_argument("--optimizer", type = str, default = "Adam")
    parser.add_argument("--stiefel_optimizer", type = str, default = "rsgd")
    parser.add_argument("--lr_euc", type = float, default = 0.01)
    parser.add_argument("--lr_stie", type = float, default = 0.1)
    parser.add_argument("--lr_scheduler", type = str, default = "step")
    parser.add_argument("--step_lr_reduce_freq", type = int, default = 500)
    parser.add_argument("--step_lr_gamma", type = float, default = 0.3)
    parser.add_argument("--lr_gamma", type =  float, default = 0.98)

    # graph parameters
    parser.add_argument("--dim", type = int, default = 16)
    parser.add_argument("--num_layers", type = int, default = 1)
    parser.add_argument("--tie_weight", action = "store_true", default = False)
    
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


def load_model_opt_sched(args, dim, K) :
    # model init
    if args.cut == 0 : 
        from gnn_pool import GNNpool
    else : 
        from gnn_pool_cc import GNNpool

    args.stie_vars = []
    args.eucl_vars = []

    model = GNNpool(args, dim, K).to(args.device)
    model.train()
    if os.path.exists("model.pt") :
        model.load_state_dict(torch.load("model.pt", map_location = args.device))
    else :
        torch.save(model.state_dict(), "model.pt")
    
    euc_opt, euc_sched, stie_opt, stie_sched = set_up_optimizer_scheduler(
        False, args, model, args.lr_euc, args.lr_stie)

    return model, euc_opt, euc_sched, stie_opt, stie_sched


def train(args, model, euc_opt, euc_sched, stie_opt, stie_sched, epochs,
          node_feats, edge_weights) :
    
    adj = torch.arange(0, node_feats.shape[0]).repeat(node_feats.shape[0], 1)
    adj.to(args.device)

    F = torch.from_numpy(node_feats).to(args.device)
    W = torch.from_numpy(edge_weights).to(args.device)

    for i in range(epochs) :
        euc_opt.zero_grad()
        stie_opt.zero_grad()

        A, S = model(F, adj, W)
        loss = model.loss(A, S)

        loss.backward()
        euc_opt.step()
        stie_opt.step()

        euc_sched.step()
        stie_sched.step()

    return S


def main() :
    extractor = ViTExtractor('dino_vits8', args.stride, model_dir = args.pretrained_weights, device = args.device)
    
    args.feature_dim = 6528 if args.log_bin else 384
    if args.mode in [1, 2] :
        foreground_k = args.K
        args.K = 2
    
    epochs = [10, 100, 10]
    
    # action
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
        # F, W are numpy arrays

        # mode 0
        model, euc_opt, euc_sched, stie_opt, stie_sched = load_model_opt_sched(args, 32, args.K)
        S = train(args, model, euc_opt, euc_sched, stie_opt, stie_sched, 
                  epochs[0], F, W)

        # polled matrix (after softmax, before argmax)
        S = S.detach().cpu()
        S = torch.argmax(S, dim = -1)

        # Post-processing Connected Component/bilateral solver
        mask0, S = graph_to_mask(S, args.cc, args.stride, image_tensor, image)
        # apply bilateral solver
        if args.bs :
            mask0 = bilateral_solver_output(image, mask0)[1]

        if args.mode == 0 :
            save_or_show([image, mask0, apply_seg_map(image, mask0, 0.7)], file, args.output_dir, args.save)
            continue

        # mode 1
        sec_index = np.nonzero(S).squeeze(1)
        F_2 = F[sec_index]
        W_2 = create_adj(F_2, args.cut, args.alpha)

        model2, euc_opt2, euc_sched2, stie_opt2, stie_sched2 = load_model_opt_sched(args, 32, foreground_k)
        S_2 = train(args, model2, euc_opt2, euc_sched2, stie_opt2, stie_sched2,
                    epochs[1], F_2, W_2)

        S_2 = S_2.detach().cpu()
        S_2 = torch.argmax(S_2, dim=-1)
        S[sec_index] = S_2 + 3

        mask1, S = graph_to_mask(S, args.cc, args.stride, image_tensor, image)

        if args.mode == 1 :
            save_or_show([image, mask1, apply_seg_map(image, mask1, 0.7)], file, args.output_dir, args.save)
            continue

        # mode 2
        sec_index = np.nonzero(S == 0).squeeze(1)
        F_3 = F[sec_index]
        W_3 = create_adj(F_3, args.cut, args.alpha)

        model3, euc_opt3, euc_sched3, stie_opt3, stie_sched3 = load_model_opt_sched(args, 32, 2)
        S_3 = train(args, model3, euc_opt3, euc_sched3, stie_opt3, stie_sched3,
                    epochs[2], F_3, W_3)

        S_3 = S_3.detach().cpu()
        S_3 = torch.argmax(S_3, dim = -1)
        S[sec_index] = S_3 + foreground_k + 5

        mask2, S = graph_to_mask(S, args.cc, args.stride, image_tensor, image)
        if args.bs :
            mask_foreground = mask0
            mask_background = np.where(mask2 != foreground_k + 5, 0, 1)
            bs_foreground = bilateral_solver_output(image, mask_foreground)[1]
            bs_background = bilateral_solver_output(image, mask_background)[1]
            mask2 = bs_foreground + (bs_background * 2)

        save_or_show([image, mask2, apply_seg_map(image, mask2, 0.7)], file, args.output_dir, args.save)
    
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
        args.K = 10

    args.manifold = LorentzManifold(args)
    args.select_manifold = "lorentz"
    args.dim = args.dim + 1
    args.weight_manifold = StiefelManifold(args, 1)
    args.proj_init = "orthogonal"

    args.device = "cuda:" + str(args.cuda) if int(args.cuda) >= 0 else "cpu"
    args.patience = args.epochs if not args.patience else int(args.patience)
    set_seed(args.seed)

    main()

    # clean-up
    if os.path.exists("model.pt") :
        os.remove("model.pt")

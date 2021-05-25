import os
import numpy as np
import torch
import random
import argparse
import pickle


def random_seed(seed_value, cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_dict', type=str, default=None)
    parser.add_argument('--dataset', help="mnist|cifar10",
                        type=str, default="cifar10")
    parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
    parser.add_argument('--dataset-mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--prune_step', type=float, default=0.1)
    parser.add_argument('--prune_percent', type=float, default=0.5)
    parser.add_argument('--global_prune', type=bool, default=True)
    parser.add_argument('--global_prune_step', type=float, default=0.1)
    parser.add_argument('--global_prune_freq', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--eita', type=float, default=0.5,
                        help="accuracy threshold")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="accuracy reduction factor")
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--train_verbose', type=bool, default=False)
    parser.add_argument('--test_verbose', type=bool, default=False)
    parser.add_argument('--prune_verbose', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--fast-dev', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    if args.args_dict is not None:
        with open(args.args_dict, 'rb') as f:
            args = torch.load(f)

    random_seed(args.seed, args.cuda)

    

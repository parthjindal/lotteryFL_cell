import os
import torch
import argparse
import pickle
from pytorch_lightning import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="mnist|cifar10",
                        type=str, default="cifar10")
    parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
    parser.add_argument('--dataset-mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--prune_step', type=float, default=0.1)
    parser.add_argument('--prune_threshold', type=float, default=0.5)
    parser.add_argument('--server_prune', type=bool, default=True)
    parser.add_argument('--server_prune_step', type=float, default=0.1)
    parser.add_argument('--server_prune_freq', type=int, default=25)
    parser.add_argument('--frac_clients_per_round', type=float, default=1.0)
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
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()

    seed_everything(seed=args.seed, workers=True)

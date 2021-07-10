import torch
import torch.nn as nn
from typing import Dict
import wandb
from torch.nn.utils import prune
from util import get_prune_summary, l1_prune, get_prune_params, copy_model, custom_save
from util import train as util_train, test as util_test


class Client():
    def __init__(
        self,
        idx,
        args,
        train_loader=None,
        test_loader=None,
        class_idxs=None,
        **kwargs
    ):
        self.idx = idx
        self.args = args
        self.test_loader = test_loader
        self.train_loader = train_loader

        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha
        self.num_data = len(self.train_loader)
        self.class_idxs = class_idxs
        self.elapsed_comm_rounds = 0

        # check if model is overfitting or not
        self.train_accuracies = []
        self.test_accuracies = []
        self.losses = []
        self.prune_rates = []
        self.cur_prune_rate = 0.00

        self.model = None
        self.global_model = None
        self.global_init_model = None
        self.global_cached_model = None

    def update(self) -> None:  # sourcery skip
        """
            Interface to Server
        """
        print(f"\n----Client:{self.idx}, IDs: {self.class_idxs}----")
        print(f"Evaluating Downloaded Subnetwork")

        if self.model is None:
            self.model = copy_model(self.global_model)
        else:
            source_params = dict(self.global_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)

        # compute metrics of downloaded model on local dataset
        metrics = self.eval(self.model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model accuracy: {accuracy}')

        _, num_zeros, num_global = get_prune_summary(model=self.global_model,
                                                     name='weight')
        prune_rate = num_zeros / num_global
        print('Global model prune percentage: {}'.format(prune_rate))

        if prune_rate < self.args.prune_threshold and accuracy >= self.eita:
            self.cur_prune_rate = self.args.prune_step
            l1_prune(model=self.model,
                     amount=self.cur_prune_rate,
                     name='weight',
                     verbose=self.args.prune_verbose)
            self.reinit(self.model, self.global_cached_model)
            self.eita = self.eita_hat

        prune_summmary, num_zeros, num_global = get_prune_summary(model=self.model,
                                                                  name='weight')
        prune_rate = num_zeros / num_global
        self.prune_rates.append(prune_rate)

        print(f"\nTraining local model")
        train_metrics = self.train(self.elapsed_comm_rounds)

        print(f"\nEvaluating Trained Model")
        metrics = self.eval(self.model)
        print(f'Trained model accuracy: {metrics["Accuracy"][0]}')

        log_dict = {f"{self.idx}_cur_prune_rate": self.cur_prune_rate,
                    f"{self.idx}_eita": self.eita,
                    f"{self.idx}_percent_pruned": self.prune_rates[-1],
                    f"{self.idx}_train_accuracy": train_metrics["Accuracy"][0]}

        for key, thing in metrics.items():
            log_dict[f"{self.idx}_{key}".lower()] = (
                thing[0] if (isinstance(thing, list)) else thing
            )

        wandb.log(log_dict)
        self.elapsed_comm_rounds += 1

    def rediscover_mask(self, model):
        params_to_prune = get_prune_params(model)
        for param, name in params_to_prune:
            amount = torch.eq(getattr(param, name),
                              0.00).sum().float()
            prune.l1_unstructured(param, name, amount=int(amount))

    def reinit(self, model, init_model):
        source_params = dict(
            init_model.named_parameters())
        for name, param in model.named_parameters():
            param.data.copy_(source_params[name].data)

    def train(self, round_index):
        """
            Train NN
        """
        losses = []
        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(
                    f"Client={self.idx}, epoch={epoch}, round:{round_index}")

            metrics = util_train(self.model,
                                 self.train_loader,
                                 self.args.lr,
                                 self.args.device,
                                 self.args.fast_dev_run,
                                 self.args.train_verbose)
            losses.append(metrics['Loss'][0])
            if self.args.fast_dev_run:
                break
        metrics = util_test(self.model,
                            self.train_loader,
                            self.args.device,
                            self.args.fast_dev_run,
                            self.args.train_verbose)
        self.losses.extend(losses)
        return metrics

    @torch.no_grad()
    def download(self, global_model, global_init_model, *args, **kwargs):
        """
            Download global model from server
        """
        self.global_model = global_model
        params_to_prune = get_prune_params(self.global_model)
        for param, name in params_to_prune:
            prune.l1_unstructured(param, name, amount=0)

        self.global_init_model = global_init_model

        params_to_prune = get_prune_params(self.global_init_model)
        for param, name in params_to_prune:
            prune.l1_unstructured(param, name, amount=0)

        if self.global_cached_model is None:
            # use this model for reinitialization

            self.global_cached_model = copy_model(
                global_init_model, self.args.device)
            metrics = util_train(
                self.global_cached_model,
                self.train_loader,
                self.args.lr,
                self.args.device,
                self.args.fast_dev_run,
                self.args.train_verbose
            )
            params_to_prune = get_prune_params(self.global_cached_model)
            for param, name in params_to_prune:
                prune.l1_unstructured(param, name, amount=0)

    def eval(self, model):
        """
            Eval self.model
        """
        eval_score = util_test(model,
                               self.test_loader,
                               self.args.device,
                               self.args.fast_dev_run,
                               self.args.test_verbose)
        self.test_accuracies.append(eval_score['Accuracy'][0])
        return eval_score

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        upload_model = copy_model(model=self.model, device=self.args.device)
        params_pruned = get_prune_params(upload_model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
        model_size = custom_save(upload_model, "/dev/null")
        return {
            'model': upload_model,
            'acc': self.test_accuracies[-1],
            'size': model_size
        }

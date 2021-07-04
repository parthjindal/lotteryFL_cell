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

    def update(self) -> None:
        """
            Interface to Server
        """
        print(f"\n----Client:{self.idx}, IDs: {self.class_idxs}----")
        print(f"Evaluating Global model")

        # compute metrics of downloaded model on local dataset
        metrics = self.eval(self.global_model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model accuracy: {accuracy}')

        _, num_zeros, num_global = get_prune_summary(model=self.global_model,
                                                     name='weight')
        prune_rate = num_zeros / num_global
        print('Global model prune percentage: {}'.format(prune_rate))

        # if local pruning threshold is not yet achieved
        if self.cur_prune_rate < self.args.prune_threshold:
            # lotteryFL condition (eita: accuracy threshold)
            if accuracy > self.eita:
                self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
                                          self.args.prune_threshold)  # inc. pruning_rate by prune_step
                # ? if downloaded model is already shallow should we prune ??
                if self.cur_prune_rate > prune_rate:
                    l1_prune(model=self.global_model,
                             amount=self.cur_prune_rate,
                             name='weight',
                             verbose=self.args.prune_verbose)
                    # reinitialize model with init_params
                    self.reinit(self.global_model, self.global_init_model)
                    # log prune %
                    self.prune_rates.append(self.cur_prune_rate)
                else:
                    # reprune by the downloaded global-model(important)
                    self.rediscover_mask(self.global_model)
                    # log prune %
                    self.prune_rates.append(prune_rate)
                # restore accuracy threshold back to original val
                self.eita = self.eita_hat
            else:
                # reprune by the downloaded global-model(important)
                self.rediscover_mask(self.global_model)
                self.eita *= self.alpha  # reduce accuracy_threshold by alpha
                # log prune %
                self.prune_rates.append(prune_rate)
        else:
            if self.cur_prune_rate > prune_rate:
                l1_prune(model=self.global_model,
                         amount=self.cur_prune_rate,
                         name='weight',
                         verbose=self.args.prune_verbose)
                # reinitialize model with init_params
                self.reinit(self.global_model, self.global_init_model)
                # log prune %
                self.prune_rates.append(self.cur_prune_rate)
            else:
                self.rediscover_mask(self.global_model)
                # log prune %
                self.prune_rates.append(prune_rate)

        self.model = self.global_model

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
            log_dict[f"{self.idx}_{key}"] = (
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
        self.global_init_model = global_init_model

        params_to_prune = get_prune_params(self.global_init_model)
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

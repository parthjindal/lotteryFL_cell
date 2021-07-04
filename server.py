import wandb
from typing import List, Dict, Tuple
import torch.nn.utils.prune as prune
import numpy as np
import os
from tabulate import tabulate
import torch
from torch.nn import Module
from util import custom_save, get_prune_params, super_prune, fed_avg, l1_prune, copy_model, get_prune_summary
from numpy import linalg as LA


class Server():
    """
        Central Server
    """

    def __init__(
        self,
        args,
        model,
        clients
    ):
        super().__init__()
        self.clients = clients
        self.num_clients = len(self.clients)
        self.args = args
        self.model = model
        self.init_model = copy_model(model, self.args.device)
        self.prev_model = copy_model(model, self.args.device)

        self.elapsed_comm_rounds = 0
        self.curr_prune_step = 0.00

        self.aggr_fn = fed_avg
        # TODO: add function switcher

    def aggr(
        self,
        models,
        clients,
        *args,
        **kwargs
    ):
        print("----------Averaging Models--------")
        weights_per_client = np.array(
            [client.num_data for client in clients], dtype=np.float32)
        weights_per_client /= np.sum(weights_per_client)

        aggr_model = self.aggr_fn(
            models=models,
            weights=weights_per_client,
            device=self.args.device
        )
        pruned_summary, _, _ = get_prune_summary(aggr_model, name='weight')
        print(tabulate(pruned_summary, headers='keys', tablefmt='github'))

        # Restore masks by manually setting zeroed params as pruned
        prune_params = get_prune_params(aggr_model)
        for param, name in prune_params:
            zeroed_weights = torch.eq(
                getattr(param, name).data, 0.00).sum().float()
            prune.l1_unstructured(param, name, int(zeroed_weights))

        return aggr_model

    def update(
        self,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """
        self.elapsed_comm_rounds += 1
        self.prev_model = copy_model(self.model, self.args.device)

        print('-----------------------------', flush=True)
        print(
            f'| Communication Round: {self.elapsed_comm_rounds}  | ', flush=True)
        print('-----------------------------', flush=True)

        _, num_pruned, num_total = get_prune_summary(self.model)
        prune_percent = num_pruned / num_total

        # global_model pruned at fixed freq
        # with a fixed pruning step
        if (self.args.server_prune == True and
            (self.elapsed_comm_rounds % self.args.server_prune_freq) == 0) and \
                (prune_percent < self.args.server_prune_threshold):
            # prune the model using super_mask
            self.prune()
            # reinitialize model
            self.reinit()

        # upload model to selected clients
        client_idxs = np.random.choice(
            self.num_clients, int(
                self.args.C*self.num_clients),
            replace=False,
        )
        clients = [self.clients[i] for i in client_idxs]
        info = self.upload(clients)
        downlink_payload = info['downlink_payload']

        # call training loop on all clients
        for client in clients:
            client.update()

        # download models from selected clients
        models, accs, uplink_payload = self.download(clients)

        avg_accuracy = np.mean(accs, axis=0, dtype=np.float32)
        print('-----------------------------', flush=True)
        print(f'| Average Accuracy: {avg_accuracy}  | ', flush=True)
        print('-----------------------------', flush=True)

        # compute average-model and (prune it by 0.00 )
        self.model = self.aggr(models, clients)

        _, num_pruned, num_total = get_prune_summary(self.model)
        prune_percent = num_pruned / num_total

        wandb.log({"Average accuracy": avg_accuracy,
                   "Communication round": self.elapsed_comm_rounds,
                   "Global pruned percentage": prune_percent,
                   "Uplink payload": uplink_payload,
                   "Downlink payload": downlink_payload})

    def prune(self):
        # all prune methods are Objects of Pruning containers hence masks build on top of each other
        if self.args.prune_method == 'l1':
            l1_prune(model=self.model,
                     amount=self.args.server_prune_step,
                     name='weight',
                     verbose=self.args.prune_verbose,
                     glob=False)
        elif self.args.prune_method == 'old_super_mask':
            super_prune(model=self.model,
                        init_model=self.init_model,
                        amount=self.args.server_prune_step,
                        name='weight',
                        verbose=self.args.prune_verbose)

    def reinit(self):
        if self.args.reinit_method == 'none':
            return

        elif self.args.reinit_method == 'std_dev':
            # reinitialize parameters based on std_dev and sign of original parameters
            source_params = dict(self.init_model.named_parameters())
            for name, param in self.model.named_parameters():
                std_dev = torch.std(source_params[name].data)
                param.data.copy_(std_dev*torch.sign(source_params[name].data))

        elif self.args.reinit_method == 'init_weights':
            # reinitialize parameters based on init_weights
            source_params = dict(self.init_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)

    def download(
        self,
        clients,
        *args,
        **kwargs
    ):
        # downloaded models are non pruned (taken care of in fed-avg)
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        uplink_payload = sum(x['size'] for x in uploads)

        return models, accs, uplink_payload

    def upload(
        self,
        clients,
        *args,
        **kwargs
    ) -> Dict:
        """
            Upload global model to clients
        """
        downlink_payload = 0  # logging
        model_size = custom_save(self.model, "/dev/null")
        downlink_payload += model_size*len(clients)

        for client in clients:
            # make pruning permanent and then upload the model to clients
            model_copy = copy_model(self.model, self.args.device)
            init_model_copy = copy_model(self.init_model, self.args.device)

            params = get_prune_params(model_copy, name='weight')
            for param, name in params:
                prune.remove(param, name)

            init_params = get_prune_params(init_model_copy)
            for param, name in init_params:
                prune.remove(param, name)
            # call client method
            client.download(model_copy, init_model_copy)

        return {
            "downlink_payload": downlink_payload
        }

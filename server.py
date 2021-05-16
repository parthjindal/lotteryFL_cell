from utils import create_model, copy_model
import random
import os
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Module
from utils import get_prune_params, average_weights_masks, evaluate, fevaluate, super_prune
import numpy as np
import torch.nn.utils.prune as prune
from typing import List, Dict, Tuple


class Server():
    """
        Central Server
    """

    def __init__(
        self,
        configs,
        test_loader,
        clients: List[Client] = [],
        comm_rounds=1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.clients = np.array(clients, dtype='object')
        self.num_clients = len(self.clients)
        self.configs = configs
        self.test_loader = test_loader
        self.init_model = create_model(configs.dataset, configs.arch)
        self.model = copy_model(self.init_model, configs.dataset, configs.arch)

    def aggr(
        self,
        models,
        *args,
        **kwargs
    ):
        return average_weights_masks(models=models,
                                     dataset=self.configs.dataset,
                                     arch=self.configs.arch,
                                     data_nums=self.num_clients)
        pass

    def update(
        self,
        prune,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """

        self.model.train()
        for i in range(self.comm_rounds):
            print('-----------------------------', flush=True)
            print(f'| Communication Round: {i+1}  | ', flush=True)
            print('-----------------------------', flush=True)
            if prune:
                self.prune(self.model)
            # broadcast model
            self.upload(self.model)
            #-------------------------------------------------#
            clients_idx = np.random.choice(
                self.num_clients, self.configs.frac * self.num_clients)
            clients = self.clients[clients_idx]
            #-------------------------------------------------#
            for client in clients():
                client.update()
            #-------------------------------------------------#
            models, accs = self.download(clients)
            self.model = self.aggr(models)

            eval_score = self.eval(self.model)
            if kwargs["verbose"] == 1:
                print(f"eval_score = {eval_score['Accuracy']}")

    def download(
        self,
        clients: List[Client],
        *args,
        **kwargs
    ):
        # TODO: parallelize downloading models from clients
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        return models, accs

    def prune(
        self,
        model,
        *args,
        **kwargs
    ):
        """
            Prune self.model
        """
        super_prune(model=model,
                    init_model=self.init_model,
                    name="weight",
                    threshold=self.configs.threshold,
                    verbose=True)

    def eval(
        self,
        model,
        *args,
        **kwargs
    ):
        """
            Eval self.model
        """
        return evaluate(model=model,
                        data_loader=self.test_loader,
                        verbose=True)

    def save(
        self,
        *args,
        **kwargs
    ):
        """
            Save model,meta-info,states
        """
        pass

    def upload(
        self,
        model,
        *args,
        **kwargs
    ) -> None:
        """
            Upload global model to clients
        """

        # TODO: parallelize upload to clients (broadcasting stratergy)
        for client in self.clients:
            model_copy = copy_model(model,
                                    self.configs.dataset,
                                    self.configs.arch
                                    )
            client.download(model_copy)

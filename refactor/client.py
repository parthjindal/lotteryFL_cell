import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from torch.nn import functional as F
import numpy as np
from typing import Dict
import copy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import math
from utils import summarize_prune
from server import Server



class Model(pl.LightningModule):
    def __init__(
        self,
        batch_size=32,
        lr=1e-3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
        self.lr = lr
        self.batch_size = batch_size,

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return {'loss': loss, 'y_hat': y_hat, 'y_true': y}

    def validation_epoch_end(self, outputs):
        correct = 0
        length = 0
        for out in outputs:
            y_hat = out["y_hat"]
            y_true = out['y_true']
            correct += y_hat.eq(y_true.view_as(y_hat)).sum().detach()
            length += len(y_hat)
        acc = correct / length
        self.log('acc', acc)
        return {'acc': acc}


model = Model()
logger = TensorBoardLogger(save_dir="./logs", name='trial')

trainer = pl.Trainer()
trainer.fit(model=model, train_dataloader=)


# class Client():
#     def __init__(
#         self,
#         idx,
#         model,
#         args,
#         train_loader,
#         test_loader,
#         **kwargs,
#     ):
#         super().__init__()

#         print("Creating model for client {}".format(idx))
#         self.idx = idx
#         self.init_model = copy.deepcopy(model)
#         self.model = model
#         self.args = args
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.elapsed_comm_rounds = 0
#         self.prune_rate = 0.00
#         self.eita = self.args.eita_hat
#         self.trainer = pl.Trainer(gpus=1)

#     def update(self) -> None:

#         print(f"-----------Training Client:{self.idx}-------")
#         # get pruning summary for globalModel
#         num_pruned, num_params = summarize_prune(
#             self.globalModel, name='weight')
#         cur_prune_rate = num_pruned / num_params
#         metrics = self.trainer.test(
#             self.globalModel, test_dataloaders=self.test_loader)
#         if metrics[0]['acc'] > self.eita:
#             self.prune_rate = min(self.prune_rate + self.args.prune_step,
#                                   self.args.prune_percent)
#             self.prune(self)

#     def Fupdate(self) -> None:
#         """
#             Interface to Server
#         """
#         print(f"----------Client:{self.client_id} Update---------------------")

#         # evaluate globalModel on local data
#         # if accuracy < eita proceed as straggler else LH finder
#         with torch.no_grad():
#             eval_score = self.eval(self.globalModel)

#             # get pruning summary for globalModel
#             num_pruned, num_params = summarize_prune(
#                 self.globalModel, name='weight')
#             cur_prune_rate = num_pruned / num_params

#             if eval_score["Accuracy"][0] > self.eita:
#                 #--------------------Lottery Finder-----------------#
#                 # expected final pruning % of local model
#                 # prune model by prune_rate - current_prune_rate
#                 # every iteration pruning should be increase by prune_step if viable
#                 self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
#                                           self.args.prune_percent)
#                 if self.cur_prune_rate > cur_prune_rate:
#                     self.prune(self.globalModel,
#                                prune_rate=self.cur_prune_rate - cur_prune_rate)
#                     self.prune_rates[self.elapsed_comm_rounds] = self.cur_prune_rate
#                     self.model = copy_model(self.global_initModel,
#                                             self.args.dataset,
#                                             self.args.arch,
#                                             source_buff=dict(self.globalModel.named_buffers()))
#                 else:
#                     self.model = self.globalModel
#                     self.prune_rates[self.elapsed_comm_rounds] = cur_prune_rate
#                 # eita reinitialized to original val
#                 self.eita = self.args.eita_hat

#             else:
#                 #---------------------Straggler-----------------------------#
#                 self.eita *= self.args.alpha
#                 self.prune_rates[self.elapsed_comm_rounds] = cur_prune_rate
#                 # copy globalModel
#                 self.model = self.globalModel
#         #-----------------------TRAINING LOOOP ------------------------#
#         # train both straggler and LH finder
#         self.model.train()
#         self.train(self.elapsed_comm_rounds)
#         self.eval_score = self.eval(self.model)

#         with torch.no_grad():
#             wandb.log(
#                 {f"{self.client_id}_cur_prune_rate": self.prune_rates[-1]})
#             wandb.log({f"{self.client_id}_eita": self.eita})

#             for key, thing in self.eval_score.items():
#                 if(isinstance(thing, list)):
#                     wandb.log({f"{self.client_id}_{key}": thing[0]})
#                 else:
#                     wandb.log({f"{self.client_id}_{key}": thing.item()})

#             if (self.elapsed_comm_rounds+1) % self.args.save_freq == 0:
#                 self.save(self.model)

#             self.elapsed_comm_rounds += 1

#     def train(self, round_index):
#         """
#             Train NN
#         """
#         accuracies = []
#         losses = []

#         for epoch in range(self.args.client_epoch):
#             train_log_path = f'./log/clients/client{self.client_id}'\
#                              f'/round{self.elapsed_comm_rounds}/'
#             if self.args.train_verbosity:
#                 print(f"Client={self.client_id}, epoch={epoch}")
#             train_score = ftrain(self.model,
#                                  self.train_loader,
#                                  self.args.lr,
#                                  self.args.train_verbosity)
#             losses.append(train_score['Loss'][-1].data.item())
#             accuracies.append(train_score['Accuracy'][-1])

#             if self.args.report_verbosity:
#                 epoch_path = train_log_path + f'client_model_epoch{epoch}.torch'
#                 epoch_score_path = train_log_path + \
#                     f'client_train_score_epoch{epoch}.pickle'
#                 log_obj(epoch_path, self.model)
#                 log_obj(epoch_score_path, train_score)

#         self.losses[round_index:] = np.array(losses)
#         self.accuracies[round_index:] = np.array(accuracies)

#     @torch.no_grad()
#     def prune(self, model, prune_rate, *args, **kwargs):
#         """
#             Prune model
#         """
#         fprune_fixed_amount(model, prune_rate,  # prune_step,
#                             verbose=self.args.prune_verbosity, glob=False)

#     @torch.no_grad()
#     def download(self, globalModel, global_initModel, *args, **kwargs):
#         """
#             Download global model from server
#         """
#         self.globalModel = globalModel
#         self.global_initModel = global_initModel

#     def eval(self, model):
#         """
#             Eval self.model
#         """
#         eval_score = fevaluate(model,
#                                self.test_loader,
#                                verbose=self.args.test_verbosity)
#         if self.args.test_verbosity:
#             eval_log_path = f'./log/clients/client{self.client_id}/'\
#                             f'round{self.elapsed_comm_rounds}/'\
#                             f'eval_score_round{self.elapsed_comm_rounds}.pickle'
#             log_obj(eval_log_path, eval_score)
#         return eval_score

#     def save(self, *args, **kwargs):
#         """
#             Save model,meta-info,states
#         """
#         if self.args.report_verbosity:
#             eval_log_path1 = f"./log/full_save/client{self.client_id}/round{self.elapsed_comm_rounds}_model.pickle"
#             eval_log_path2 = f"./log/full_save/client{self.client_id}/round{self.elapsed_comm_rounds}_dict.pickle"
#             log_obj(eval_log_path1, self.model)
#             log_obj(eval_log_path2, self.__dict__)

#     def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
#         """
#             Upload self.model
#         """
#         return {
#             "model": copy_model(self.model,
#                                 self.args.dataset,
#                                 self.args.arch),
#             "acc": self.eval_score["Accuracy"]
#         }

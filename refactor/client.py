import pytorch_lightning as pl
from utils import copy_model, create_model, L1Prune, get_prune_params
from torch.nn.utils import prune
from typing import Dict
from torch.utils.data import DataLoader
from utils import Test, Train
from tabulate import tabulate


class Client():
    def __init__(
        self,
        idx,
        args,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        **kwargs
    ):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.prune_threshold = args.prune_threshold
        self.prune_step = args.prune_step
        self.args = args
        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha

        self.model = None
        self.global_model = None
        self.global_init_model = None
        self.elapsed_comm_rounds = 0

        ### TODO ###
        self.num_data = len(self.train_loader)
        self.curr_prune_rate = 0.0
        self.accuracy = 0.0
        self.logger = None
        ######## TODO : implement logger class ########
        # just add prefix to the global logger using the client id
        # and implement the log function

    def update(self):
        self.elapsed_comm_rounds += 1
        metrics = Test(model=self.global_model,
                       test_dataloaders=self.test_loader,
                       device=self.args.device,
                       fast_dev_run=self.args.fast_dev_run,
                       verbose=self.args.test_verbose)

        accuracy = metrics["Accuracy"]
        print("Global model accuracy: {}".format(accuracy))

        if self.curr_prune_rate < self.prune_threshold:
            if accuracy > self.eita:
                self.curr_prune_rate = min(self.curr_prune_rate + self.prune_step,
                                           self.prune_threshold)

                L1Prune(self.global_model, self.curr_prune_rate,
                        'weight', verbose=self.args.prune_verbose)
                Client.Reinit(self.global_model, self.global_init_model)
                self.model = copy_model(self.global_model)
                self.eita = self.eita_hat

            else:
                self.eita *= self.alpha
                self.model = copy_model(self.global_model)

        else:
            L1Prune(self.global_model, self.curr_prune_rate,
                    'weight', verbose=self.args.prune_verbose)
            self.model = copy_model(self.global_model)

        train(self.model)

        # TODO: log training log data
        metrics = Test(model=self.model,
                       test_dataloaders=self.test_loader,
                       device=self.args.device,
                       fast_dev_run=self.args.fast_dev_run,
                       verbose=self.args.test_verbose)

        self.accuracy = metrics['Accuracy']
        print("Trained model accuracy: {}".format(self.accuracy))

    def train(self, model):
        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(f"Client={self.client_id}, epoch={epoch}")
            metrics = Train(model,
                            self.train_loader,
                            self.args.lr,
                            self.args.device,
                            self.args.fast_dev_run,
                            self.args.train_verbose)
            if self.args.train_verbose:
                print(tabulate(metrics, headers='keys', tablefmt='github'))

            # if self.args.log:
            #     self.logger.log(metrics)

    @staticmethod
    def Reinit(model, init_model):
        print("Reinitializing the model")
        source_params = dict(init_model.named_parameters())
        for name, param in model.named_parameters():
            param.data.copy_(source_params[name].data)

    def download(self, model, init_model, *args, **kwargs):
        """
            Download global model from server
        """
        self.global_model = model
        self.global_init_model = init_model

    def upload(self) -> Dict:
        """
            Uploads a model copy,accuracy
            Note: The uploaded model has any/all prunings made permanent
        """
        upload_model = copy_model(model=self.model)
        params_pruned = get_prune_params(upload_model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
        return {
            'model': upload_model,
            'acc': self.accuracy
        }

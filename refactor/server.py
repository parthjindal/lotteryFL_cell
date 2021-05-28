import numpy as np
from utils import copy_model, SuperPrune, fed_avg, L1Prune


class Server():
    def __init__(
        self,
        args,
        model,
        clients,
    ):
        self.clients = clients
        self.num_clients = len(self.clients)
        self.args = args
        self.elapsed_comm_rounds = 0
        self.total_comm_rounds = args.comm_rounds
        self.model = model
        self.init_model = copy_model(model)
        self.client_accs = np.zeros((self.num_clients, args.comm_rounds))

    def aggr(
        self,
        models,
        clients
    ):
        weights_per_client = np.array(
            [client.num_data for client in clients], dtype=np.float32)
        weights_per_client /= np.sum(weights_per_client)
        aggr_model = fed_avg(
            models=models,
            weights=weights_per_client
        )
        L1Prune(aggr_model, amount=0.0, name='weight')
        return aggr_model

    def update(
        self,
    ):
        self.elapsed_comm_rounds += 1
        print('-----------------------------', flush=True)
        print(
            f'| Communication Round: {self.elapsed_comm_rounds}  | ', flush=True)
        print('-----------------------------', flush=True)

        if (self.args.server_prune == True and
                (self.elapsed_comm_rounds % self.args.server_prune_freq) == 0):
            SuperPrune(
                model=self.model,
                init_model=self.init_model,
                amount=self.args.server_prune_step,
                name='weight'
            )
            Server.Reinit(dest_model=self.model,
                          source_model=self.init_method)

        client_idxs = np.random.choice(
            self.num_clients, int(
                self.args.frac_clients_per_round*self.num_clients),
            replace=False,
        )
        clients = [self.clients[i] for i in client_idxs]
        self.upload(clients)

        for client in clients:
            client.update()

        models, accs = self.download(clients)

        print("Average Client accuracy: {}".format(np.mean(accs, axis=0)))

        aggr_model = self.aggr(models, clients)
        Server.Reinit(
            dest_model=self.model,
            source_model=aggr_model)

    def download(
        self,
        clients
    ):
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        return models, accs

    def upload(
        self,
        clients
    ):
        for client in clients:
            init_model_copy = copy_model(model=self.init_model)
            model_copy = copy_model(model=self.model)
            client.download(model_copy, init_model_copy)

    @staticmethod
    def Reinit(dest_model, source_model):
        """

        """
        source_params = dict(source_model.named_parameters())
        for name, param in dest_model.named_parameters():
            param.data.copy_(source_params[name].data)

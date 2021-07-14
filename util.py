import os
import sys
import errno
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as skmetrics
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
import torch.nn.utils.prune as prune
import io
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from torch.nn import functional as F
import gzip


@torch.no_grad()
def fed_avg(models: List[nn.Module], weights: torch.Tensor, device='cuda:0'):
    """
        models: list of nn.modules(unpruned/pruning removed)
        weights: normalized weights for each model
        cls:  Class of original model
    """
    aggr_model = models[0].__class__().to(device)
    num_models = len(models)
    model_params = [dict(model.named_parameters()) for model in models]

    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        for i in range(num_models):
            weighted_param = torch.mul(
                model_params[i][name].data, weights[i])
            param.data.copy_(param.data + weighted_param)
    return aggr_model


@torch.no_grad()
def merge_models(models: List[nn.Module], weights: torch.Tensor, device='cuda:0'):
    """
        Aggregates the model by merging subnetworks ie. averaged by no. of 
        non-zero weight vals across subnetworks

        models: list of nn.modules(unpruned/pruning removed)
        weights: normalized weights for each model
        cls:  Class of original model

    """
    aggr_model = models[0].__class__().to(device)
    num_models = len(models)
    model_params = [dict(model.named_parameters()) for model in models]

    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        non_zero_weights = torch.ones_like(param.data)*(1E-6)
        for i in range(num_models):
            weighted_param = model_params[i][name].data
            non_zero_weights += ~torch.eq(weighted_param, 0.00)
            param.data.copy_(param.data + weighted_param)
        param.data.copy_(torch.div(param.data, non_zero_weights))
    return aggr_model


def create_model(cls, device='cuda:0') -> nn.Module:
    """
        Returns new model pruned by 0.00 %. This is necessary to create buffer masks
    """
    model = cls().to(device)
    l1_prune(model, amount=0.00, name='weight', verbose=False)
    return model


def copy_model(model: nn.Module, device='cuda:0'):
    """
        Returns a copy of the input model.
        Note: the model should have been pruned for this method to work to create buffer masks and what not.
    """
    new_model = create_model(model.__class__, device)
    source_params = dict(model.named_parameters())
    source_buffer = dict(model.named_buffers())
    for name, param in new_model.named_parameters():
        param.data.copy_(source_params[name].data)
    for name, buffer_ in new_model.named_buffers():
        buffer_.data.copy_(source_buffer[name].data)
    return new_model


metrics = MetricCollection([
    Accuracy(),
    Precision(),
    Recall(),
    F1(),
])


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    lr: float = 1e-3,
    device: str = 'cuda:0',
    fast_dev_run=False,
    verbose=True
) -> Dict[str, torch.Tensor]:

    optimizer = optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    num_batch = len(train_dataloader)
    global metrics

    metrics = metrics.to(device)
    model.train(True)
    torch.set_grad_enabled(True)

    losses = []
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=num_batch,
                        disable=not verbose,
                        )

    for batch_idx, batch in progress_bar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = F.softmax(y_hat, 1)
        output = metrics(preds, y)

        progress_bar.set_postfix({'loss': loss.item(),
                                  'acc': output['Accuracy'].item()})
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    outputs = {k: [v.item()] for k, v in outputs.items()}
    outputs['Loss'] = [sum(losses) / len(losses)]
    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


@ torch.no_grad()
def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    device='cuda:0',
    fast_dev_run=False,
    verbose=True,
) -> Dict[str, torch.Tensor]:

    num_batch = len(test_dataloader)
    model.eval()
    global metrics

    metrics = metrics.to(device)
    progress_bar = tqdm(enumerate(test_dataloader),
                        total=num_batch,
                        file=sys.stdout,
                        disable=not verbose)
    for batch_idx, batch in progress_bar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        preds = F.softmax(y_hat, 1)
        output = metrics(preds, y)

        progress_bar.set_postfix({'acc': output['Accuracy'].item()})
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    model.train(True)
    outputs = {k: [v.item()] for k, v in outputs.items()}

    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


def l1_prune(model, amount=0.00, name='weight', verbose=False, glob=False):
    """
        Prunes the model param by param by given amount
    """
    params_to_prune = get_prune_params(model, name)
    if glob:
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount)
    else:
        for params, name in params_to_prune:
            prune.l1_unstructured(params, name, amount)
    if verbose:
        info, num_zeros, num_global = get_prune_summary(model, name)
        global_pruning = num_zeros / num_global
        print(tabulate(info, headers='keys', tablefmt='github'))
        print("Total Pruning: {}%".format(global_pruning))


"""
Hadamard Mult of Mask and Attributes,
then return zeros
"""


@ torch.no_grad()
def summarize_prune(model: nn.Module, name: str = 'weight') -> tuple:
    """
        returns (pruned_params,total_params)
    """
    num_pruned = 0
    params, num_global_weights, _ = get_prune_params(model)
    for param, _ in params:
        if hasattr(param, name+'_mask'):
            data = getattr(param, name+'_mask')
            num_pruned += int(torch.sum(data == 0.0).item())
    return (num_pruned, num_global_weights)


def get_prune_params(model, name: str = 'weight') -> List[Tuple[nn.Parameter, str]]:

    params_to_prune = []
    if hasattr(model, 'prunable_modules'):
        for mod_name, module in model.named_modules():
            if mod_name in model.prunable_modules and hasattr(module, name):
                params_to_prune.append((module, name))
        return params_to_prune

    else:
        for _, module in model.named_children():
            for name_, param in module.named_parameters():
                if name in name_:
                    params_to_prune.append((module, name))

    return params_to_prune


def get_prune_summary(model, name='weight') -> Tuple[Union[Union[Dict[str, Union[List[Union[str, float]], float]], int], int]]:
    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    num_global_weights = 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}
    params_pruned = get_prune_params(model, 'weight')

    for layer, weight_name in params_pruned:

        num_layer_zeros = torch.sum(
            getattr(layer, weight_name) == 0.0).item()
        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        num_global_weights += num_layer_weights
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(layer.__str__())
        prune_stat['Weight Name'].append(weight_name)
        prune_stat['Percent Pruned'].append(
            f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

    global_prune_percent = num_global_zeros / num_global_weights

    return prune_stat, num_global_zeros, num_global_weights


def custom_save(model, path) -> int:
    """
    https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
    Custom save utility function
    Compresses the model using gzip
    Helpful if model is highly pruned

    Returns compressed model_size
    """
    bufferIn = io.BytesIO()
    torch.save(model.state_dict(), bufferIn)
    bufferOut = gzip.compress(bufferIn.getvalue())
    bufferLen = len(bufferOut) / (1024.0)**2  # size in MB
    with gzip.open(path, 'wb') as f:
        f.write(bufferOut)
    return bufferLen


def custom_load(path) -> Dict:
    """
    returns saved_dictionary
    """
    with gzip.open(path, 'rb') as f:
        bufferIn = f.read()
        bufferOut = gzip.decompress(bufferIn)
        state_dict = torch.load(io.BytesIO(bufferOut))
    return state_dict


def log_obj(path, obj):
    # pass
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #
    with open(path, 'wb') as file:
        if isinstance(obj, nn.Module):
            torch.save(obj, file)
        else:
            pickle.dump(obj, file)


class CustomPruneMethod(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount, orig_weights):
        super().__init__()
        self.amount = amount
        self.original_signs = self.get_signs_from_tensor(orig_weights)

    def get_signs_from_tensor(self, t: torch.Tensor):
        return torch.sign(t).view(-1)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        large_weight_mask = t.view(-1).mul(self.original_signs)
        large_weight_mask_ranked = F.relu(large_weight_mask)
        nparams_toprune = int(torch.numel(t) * self.amount)  # get this val
        if nparams_toprune > 0:
            bottom_k = torch.topk(
                large_weight_mask_ranked.view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[bottom_k.indices] = 0.00
        return mask


def customPrune(module, orig_module, amount=0.1, name='weight'):
    """
        Taken from https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        Takes: current module (module), name of the parameter to prune (name)

    """
    CustomPruneMethod.apply(module, name, amount, orig_module)
    return module


def super_prune(
    model: nn.Module,
    init_model: nn.Module,
    amount: float = 0.0,
    name: str = 'weight',
    verbose=False
) -> None:
    """

    """
    params_to_prune = get_prune_params(model)
    init_params = get_prune_params(init_model)

    for idx, (param, name) in enumerate(params_to_prune):
        orig_params = getattr(init_params[idx][0], name)

        # original params are sliced by the pruned model's mask
        # this is because pytorch's pruning container slices the mask by
        # non-zero params
        if hasattr(param, 'weight_mask'):
            mask = getattr(param, 'weight_mask')
            sliced_params = orig_params[mask.to(torch.bool)]
            customPrune(param, sliced_params, amount, name)
        else:
            customPrune(param, orig_params, amount, name)

    if verbose:
        info, num_zeros, num_global = get_prune_summary(model, name)
        global_pruning = num_zeros / num_global
        print(tabulate(info, headers='keys', tablefmt='github'))
        print("Total Pruning: {}%".format(global_pruning))

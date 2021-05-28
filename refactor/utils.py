import io
import gzip
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from torch.nn.utils import prune
from torch.nn import functional as F
from pytorch_lightning.metrics import functional as FM
from tqdm import tqdm
import sys
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
from tabulate import tabulate


def get_prune_params(model, name) -> List[Tuple[nn.Parameter, str]]:
    params_to_prune = []
    for _, module in model.named_children():
        for name_, param in module.named_parameters():
            if name in name_:
                params_to_prune.append((module, name))
    return params_to_prune


def L1Prune(model, amount=0.00, name='weight', verbose=True):
    """
        Prunes the model param by param by given amount
    """
    params_to_prune = get_prune_params(model, name)
    for params, name in params_to_prune:
        prune.l1_unstructured(params, name, amount)
    if verbose:
        info = get_prune_summary(model, name)
        
        print(tabulate())


def create_model(cls) -> nn.Module:
    """
        Returns new model pruned by 0.00 %. This is necessary to create buffer masks
    """
    model = cls()
    L1Prune(model, amount=0.00, name='weight')
    return model


def copy_model(model: nn.Module):
    """
        Returns a copy of the input model.
        Note: the model should have been pruned for this method to work to create buffer masks and what not.
    """
    new_model = create_model(model.__class__)
    source_params = dict(model.named_parameters())
    source_buffer = dict(model.named_buffers())
    for name, param in new_model.named_parameters():
        param.data.copy_(source_params[name].data)
    for name, buffer_ in new_model.named_buffers():
        buffer_.data.copy_(source_buffer[name].data)
    return new_model


def custom_save(model, path):
    """
    https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
    Custom save utility function
    Compresses the model using gzip
    Helpfull if model is highly pruned
    """
    bufferIn = io.BytesIO()
    torch.save(model.state_dict(), bufferIn)
    bufferOut = gzip.compress(bufferIn.getvalue())
    with gzip.open(path, 'wb') as f:
        f.write(bufferOut)


def custom_load(path) -> Dict:
    """
    returns saved_dictionary
    """
    with gzip.open(path, 'rb') as f:
        bufferIn = f.read()
        bufferOut = gzip.decompress(bufferIn)
        state_dict = torch.load(io.BytesIO(bufferOut))
    return state_dict


@torch.no_grad()
def fed_avg(models: List[nn.Module], weights: torch.Tensor):
    """
        models: list of nn.modules(unpruned/pruning removed)
        weights: normalized weights for each model
        cls:  Class of original model
    """
    aggr_model = models[0].__class__()
    model_params = []
    num_models = len(models)

    for model in models:
        model_params.append(dict(model.named_parameters()))

    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        for i in range(num_models):
            weighted_param = torch.mul(
                model_params[i][name].data, weights[i])
            param.data.copy_(param.data + weighted_param)
    return aggr_model


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
        else:
            return mask


def customPrune(module, orig_module, amount=0.1, name='weight'):
    """
        Taken from https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        Takes: current module (module), name of the parameter to prune (name)

    """
    CustomPruneMethod.apply(module, name, amount, orig_module)
    return module


def SuperPrune(
    model: nn.Module,
    init_model: nn.Module,
    amount: float = 0.0,
    name: str = 'weight'
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


def get_prune_summary(model, name='weight') -> Dict[str, Union[List[Union[str, float]], float]]:
    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    num_global_weights = 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}
    params_pruned = get_prune_params(model, 'weight')

    for param, name in params_pruned:
        try:
            mask = getattr(param, name+'_mask')
            num_layer_zeros = torch.eq(mask, 0.0).sum().item()
        except Exception:
            num_layer_zeros = 0

        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(param, name))

        num_global_weights += num_layer_weights
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(param.__str__())
        prune_stat['Weight Name'].append(name)
        prune_stat['Percent Pruned'].append(
            f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

        global_prune_percent = num_global_zeros / num_global_weights

    prune_stat['global'] = global_prune_percent
    return prune_stat


metrics = MetricCollection([
    Accuracy(),
    Precision(num_classes=10, average='macro'),
    Recall(num_classes=10, average='macro'),
    F1(num_classes=10, average='macro'),
])


def Train(
    model: nn.Module,
    train_dataloader: DataLoader,
    lr: float = 1e-3,
    device: str = 'cuda:0',
    fast_dev_run=False,
    verbose=True
) -> Dict[str, torch.Tensor]:

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    num_batch = len(train_dataloader)
    global metrics

    metrics = metrics.to(device)
    model.train(True)
    torch.set_grad_enabled(True)

    losses = []
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=num_batch,
                        disable=not verbose)

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
        output = metrics(y_hat, y)

        progress_bar.set_postfix({'loss': loss.item(),
                                  'acc': output['Accuracy'].item()})
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    torch.set_grad_enabled(False)
    outputs['Loss'] = sum(losses) / len(losses)

    return outputs


@ torch.no_grad()
def Test(
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

        output = metrics(y_hat, y)

        progress_bar.set_postfix({'acc': output['Accuracy'].item()})
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    model.train(True)

    return outputs

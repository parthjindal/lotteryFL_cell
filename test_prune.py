from __future__ import print_function
import time
from matplotlib import pyplot
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import super_prune
from utils import prune_fixed_amount, get_prune_params
from torch.nn.utils import prune
from provided_code.datasource import get_data, DataLoaders


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


orig_model = Net().to(device="cuda:0")
orig_model.load_state_dict(torch.load("mnist_orig.pt"))

trained_model1 = Net().to(device="cuda:0")
trained_model1.load_state_dict(torch.load("mnist_cnn.pt"))

trained_model2 = Net().to(device="cuda:0")
trained_model2.load_state_dict(torch.load("mnist_cnn.pt"))

use_cuda = True
batch_size = 32


test_kwargs = {'batch_size': batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# dataset2 = datasets.MNIST('./data', train=False,
#                           transform=transform)
# test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
_, test_loaders, test_loader_ = DataLoaders(num_users=1,
                                           dataset_name='mnist',
                                           n_class=2,
                                           nsamples=100,
                                           mode='non-iid',
                                           batch_size=32,
                                           rate_unbalance=1.0)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    length = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # print(pred,target.view_as(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()
            length += len(data)

    test_loss /= length

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, length,
        100. * correct / length))


def reinit(model, source_buff=None):
    new_model = Net().to(device="cuda:0")
    prune_fixed_amount(new_model, amount=0.0, verbose=False)
    source_weights = dict(model.named_parameters())
    source_buffers = source_buff
    for name, param in new_model.named_parameters():
        if 'weight' in name:
            std = torch.std(source_weights[name])
            param.data.copy_(torch.sign(source_weights[name])*std)
        else:
            param.data.copy_(source_weights[name])
    for name, buffer in new_model.named_buffers():
        buffer.data.copy_(source_buffers[name])
    return new_model


prune_fixed_amount(orig_model, amount=0.00, verbose=False)
# prune_fixed_amount(trained_model2, amount=0.45, verbose=True, glob=False)
super_prune(trained_model2, orig_model, name='weight',
            threshold=0.2, verbose=True)

pruned_model = reinit(orig_model, source_buff=dict(
    trained_model2.named_buffers()))


# torch.save(pruned_model.state_dict(),"pruned_model.pt")


test(trained_model1, "cuda:0", test_loaders[0])
test(pruned_model, "cuda:0", test_loaders[0])
test(orig_model, "cuda:0", test_loaders[0])

# trained_model1()

for i,(x,y) in enumerate(test_loaders[0]):
    # define subplot
    x = x.to(device = "cuda:0")
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    print(y[0])
    print(trained_model1(x[0:1]).argmax(dim=1, keepdims=True))
    pyplot.imshow(x.cpu()[0].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    if i == 3:
        break
# show the figure
pyplot.show()
time.sleep(0)

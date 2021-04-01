import os
import numpy as np
import pandas as pd
import argparse

#pytorch
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms

import Models
import Trainer
import utils

args = utils.parser.parse_args()

"""
describe argument

data_dir = '../chest_xray' --dataset path for training
log_path = 'log' --path for logging loss
log_metrics = True  --log metrics(True) or not(False)
device_number = 1 --gpu number
channel = 1 --number of channels of input image. It must be 1 or 3
mode = 'auto' --select model you want (auto or cnn)
lr = 0.0001 --learning rate
batch_size = 32 -- batch size
workers = 4 --workers for dataloader
num_epoch = 30 --epoch
layers = 50 --number of resnet layers. It must be 50 or 101 or 152
classes = 2  --number of classes you want to classify
"""

if args.channel == 1:
    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'val' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
    }
elif args.channel == 3:
    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]),
        'val' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    }

assert args.channel in (1,3), "input channel must be 1 or 3"


model_mode = True if args.mode == 'auto' else False
use_cuda = torch.cuda.is_available()
device = torch.device(f'cuda:{args.device_number}' if use_cuda else 'cpu')
if use_cuda:
    torch.cuda.set_device(device)


image_datasets = {x:datasets.ImageFolder(os.path.join(args.data_dir, x),
                                        data_transforms[x])
                                        for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size = args.batch_size,
                                              shuffle=True,
                                              num_workers = args.workers)
                                              for x in ['train', 'val']}

dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}

if args.resnet_layers == 50:
    layers = [3,4,6,3]
elif args.resnet_layers == 101:
    layers = [3,4,23,3]
elif args.resnet_layers == 152:
    layers = [3,8,36,3]
assert args.resnet_layers in (50, 101, 152), "resnet_layers must be 50 or 101 or 152"


models = Models.AutoNet(
    block = Models.Bottleneck,
    layers = layers,
    in_channels = args.channel,
    out_channels = args.channel,
    auto = model_mode,
    num_classes = args.classes
)

MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()
criterions = [MSE, CE]
optimizers = torch.optim.Adam(models.parameters(), lr = args.lr)

trained = Trainer.train_model(
    models = models,
    dataloaders = dataloaders,
    dataset_sizes = dataset_sizes,
    criterions = criterions,
    optimizers = optimizers,
    num_epoch = args.num_epoch,
    device = device
)

classifier, best_acc, train_loss_history, val_loss_history = trained
classifier.eval()
torch.save(classifier, os.path.join(args.log_path, 'classifier.pytorch'))

if args.log_metrics:
    df_train = pd.DataFrame(train_loss_history)
    df_train.to_csv(os.path.join(args.log_path, 'df_train.csv'))

    df_val = pd.DataFrame(val_loss_history)
    df_val.to_csv(os.path.join(args.log_path, 'df_val.csv'))

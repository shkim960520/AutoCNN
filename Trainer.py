import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(models, dataloaders, dataset_sizes, criterions, optimizers, num_epoch, device):
    #log loss
    train_loss_history = []
    val_loss_history = []

    #model
    classifier = models
    classifier = classifier.to(device)

    #loss function
    mse, ce = criterions

    #optimizers
    optimizer = optimizers

    since = time.time()

    #copy best state
    best_model_wts = copy.deepcopy(classifier.state_dict())
    best_acc = 0.0

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch-1))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                classifier.train()
            else:
                classifier.eval()

            running_cls_loss = 0.0
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                y_hat, x_hat = classifier(inputs)

                _, preds = torch.max(y_hat, 1)
                cls_loss = ce(y_hat, labels)

                loss = cls_loss
                if x_hat != None:
                    rec_loss = mse(x_hat, inputs)
                    loss = cls_loss + rec_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_cls_loss += cls_loss.item() * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_cls_loss = running_cls_loss / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_history.append(epoch_cls_loss)
            else:
                val_loss_history.append(epoch_cls_loss)


            print('{} total: {:.4f} cls: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_cls_loss, epoch_acc
            ))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(classifier.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    classifier.load_state_dict(best_model_wts)


    return classifier, best_acc, train_loss_history, val_loss_history

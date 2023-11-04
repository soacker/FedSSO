import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np


class client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, device, id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate, local_steps):
        self.model = copy.deepcopy(model)
        self.device = device
        self.id = id  # integer
        self.train_slow = train_slow
        self.send_slow = send_slow
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_steps = local_steps

        self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False, shuffle=True)
        # self.trainloader = DataLoader(train_data, self.batch_size, drop_last=True)
        self.testloader = DataLoader(test_data, self.batch_size, drop_last=False)
        self.trainloaderfull = DataLoader(train_data, self.batch_size, drop_last=False, shuffle=True)
        self.testloaderfull = DataLoader(test_data, self.batch_size, drop_last=False)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)



    def set_parameters(self, model, epoch):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_paramenters(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_accuracy(self):
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in self.testloaderfull:
                y = y.flatten()
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                test_num += y.shape[0]
        
        return test_acc, test_num


    def train_accuracy_and_loss(self):
        self.model.eval()

        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in self.trainloaderfull:
            # x = self.one_hot_encode(x)
            y = y.flatten()

            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)

            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()


            train_num += y.shape[0]
            loss += self.loss(output, y).item()

        return train_acc, loss, train_num

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (x, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (x, y) = next(self.iter_trainloader)

        return (x.to(self.device), y.to(self.device))


    def one_hot_encode(self, arr, n_labels=65):
        # Initialize the the encoded array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.int().flatten()] = 1.

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return torch.from_numpy(one_hot)

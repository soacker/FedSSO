import copy

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from flcore.clients.clientbase import client
import numpy as np
import time


class clientAVG(client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)

        self.loss = nn.CrossEntropyLoss()
        # self.loss = torch.nn.BCEWithLogitsLoss()
        # self.loss = nn.BCELoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
        #                                  momentum=0.5, weight_decay=5e-4)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                         momentum=0, weight_decay=0)


    def train(self, epoch=0):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            # for idx, (x, y) in enumerate(self.trainloader):
            #     # x, y = self.get_next_train_batch()
            #
            #     # x = self.one_hot_encode(x)
            #     y = y.flatten()
            #     x = x.to(self.device)
            #     y = y.to(self.device)
            #
            #     self.optimizer.zero_grad()  # zero the gradient buffer
            #     # loss, accuracy, precision, recall, f1 = self.compute_loss(x, y)
            #     predictions = self.model(x)
            #     loss = self.loss(predictions, y)
            #     loss.backward()
            #     self.optimizer.step()
            #
            #     # break


            x, y = self.get_next_train_batch()
            y = y.flatten()
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.loss(predictions, y)
            loss.backward()
            self.optimizer.step()



        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def show_base(self, xx, epoch):

        if epoch % 50 == 0:
            plt.imshow(xx.reshape(28, 28, 1))
            plt.show()

            xx = torch.flatten(xx, 1)
            xx = self.model.fn_base(xx)

            plt.imshow(xx.reshape(28, 28, 1))
            plt.show()

    def train_accuracy_and_loss(self):
        # self.model.to(self.device)
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

            # 展示base
            # self.show_base(x[0])

            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # train_acc += (torch.sum(torch.argmax(output, dim=1) == y.flatten())).item()
            # train_acc += torch.sum(output.flatten() == y.flatten()).item()


            train_num += y.shape[0]
            # loss += self.loss(output, y).item() * y.shape[0]

            # loss += self.loss(output, y.view(len(y), 1).float()).item()
            # loss += self.loss(output, y.flatten()).item()

            # loss += self.loss(output.flatten(), y.float()).item()
            loss += self.loss(output, y).item()

            # torch.nn.functional.binary_cross_entropy_with_logits(output, y, reduction="mean")
            # loss += self.loss(output, y.reshape(len(y), 1)).item() * y.shape[0]


        return train_acc, loss, train_num

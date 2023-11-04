import copy
import time

import torch

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_data, read_client_data
from threading import Thread


class server_adam_paper_2023(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps,
                 num_clients,
                 total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal,
                 time_threthold, config=None):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select,
                         goal,
                         time_threthold)
        # select slow clients
        self.set_slow_clients()

        for i, train_slow, send_slow in zip(range(self.total_clients), self.train_slow_clients, self.send_slow_clients):
            train, test = read_client_data(dataset, i)
            # train, test = read_data(dataset, i)
            client = clientAVG(device, i, train_slow, send_slow, train,
                               test, model, batch_size, learning_rate, local_steps)
            self.clients.append(client)

        print(
            f"Number of clients / total clients: {self.num_clients} / {self.total_clients}")
        print("Finished creating server and clients.")

        self.x_k_1 = copy.deepcopy(self.global_model)
        self.x_k = copy.deepcopy(self.global_model)

        self.model_list = []
        self.opimzer_list = []
        for para in list(self.x_k.parameters()):
            self.model_list.append(para)

        self.split_layer = [0, 59]

        self.model_lengths = []
        self.select_layers = []
        # for layer, param in self.global_model.state_dict().items():
        for layer, param in self.global_model.named_parameters():
            param_len = self.parameter_length(param.data.size())
            if param_len < 1000000 and param_len > 0:
                self.select_layers.append(layer)
                self.model_lengths.append(param_len)
            else:
                print("over length param_len")

        select_layers = self.select_layers[0:]
        model_length = sum(self.model_lengths[0:])
        self.opimzer_list.append(
            Optimizer_adam(self.global_model, 0, len(select_layers), model_length,
                           select_layers, device, learning_rate, config=config))

    @staticmethod
    def parameter_length(size: torch.Size):
        if len(size) >= 5:
            raise TypeError
        l = torch.prod(torch.tensor(size)).item()
        if l != int(l):
            raise TypeError
        return int(l)

    def train(self):
        for i in range(self.global_rounds + 1):

            s_t = time.time()

            self.x_k_1 = copy.deepcopy(self.global_model)
            self.send_models(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.selected_clients = self.select_clients()

            for client in self.selected_clients:
                client.train(i)

            self.receive_models()
            self.aggregate_parameters()
            self.x_k = copy.deepcopy(self.global_model)

            self.model_list = []
            for optimizer in self.opimzer_list:
                x_k = optimizer.update_model(epoch=i, x_k_1=self.x_k_1, x_k=self.x_k)
                self.model_list.extend(x_k)
            for global_model_param, para in zip(self.global_model.parameters(), self.model_list):
                global_model_param.data = para.data.clone()
            # for model_param in self.opimzer_list:

            # print(list(self.x_k.parameters()))
            # print(list(self.global_model.parameters()))

            print('-' * 25, 'client trains time cost', '-' * 25, time.time() - s_t)

            if self.stop_sigma:
                print("stop sigma: on cifar100")
                break

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()


class Optimizer_SSO(object):
    def __init__(self, model, from_layer, to_layer, model_length, select_layers, device, learning_rate):
        self.model = copy.deepcopy(model)
        self.from_layer = from_layer
        self.to_layer = to_layer

        self.model_length = model_length
        self.select_layers = select_layers

        self.g_k_2 = 0
        self.g_k_1 = 0
        self.x_k_2 = 0
        self.x_k_1 = 0
        self.x_k = 0
        self.D_k_2 = 0

        self.device = device

        self.learning_rate = learning_rate

        self.delta = None

    @staticmethod
    def parameter_length(size: torch.Size):
        if len(size) >= 5:
            raise TypeError
        l = torch.prod(torch.tensor(size)).item()
        if l != int(l):
            raise TypeError
        return int(l)

    def transfer_model_to_matrix(self, model):
        param_dict = model.state_dict()
        select_param_list = []
        for layer in self.select_layers:
            param = param_dict[layer]
            x1 = param.resize(self.parameter_length(param.data.size()), )
            # print("add platen tensor {} -> {}".format(list(param.data.size()), list(x1.data.size())))
            select_param_list.append(x1)

        x = torch.cat(tensors=select_param_list, dim=0)
        assert (x.size()[0] == self.model_length)
        return x

    # 将一维还原为多个参数列表
    def transfer_matrix_to_model(self, matrix):
        param_dict = self.model.state_dict()
        idx = 0
        m = []
        for layer in self.select_layers:
            shape = param_dict[layer].data.size()
            param_len = self.parameter_length(shape)
            x1 = matrix[idx: idx + param_len].reshape(shape)
            # print("restore tensor {}({}) -> {}".format(param_len, list(shape), list(x1.data.size())))
            idx += param_len

            m.append(x1)

        assert (idx == matrix.size()[0])
        # print("restore param count {} {}".format(idx, matrix.size()[0]))
        return m

    def update_model(self, epoch, x_k_1, x_k):
        if epoch == 0:
            self.x_k_1 = 0
            self.g_k_1 = 0
            self.D_k_2 = torch.eye(self.model_length).to(self.device)

        # 根据DFP
        # 替换

        self.x_k_2 = self.x_k_1
        self.x_k_1 = self.transfer_model_to_matrix(x_k_1).reshape(self.model_length, 1)
        self.x_k = self.transfer_model_to_matrix(x_k).reshape(self.model_length, 1)

        # del pre_model

        self.g_k_2 = self.g_k_1
        self.g_k_1 = (self.x_k_1 - self.x_k) / (self.learning_rate)  # * self.local_steps

        sk_sigma = 0.7
        s_k_2 = (self.x_k_1 - self.x_k_2).reshape(self.model_length, 1) * sk_sigma  #
        y_k_2 = (self.g_k_1 - self.g_k_2).reshape(self.model_length, 1)

        lambda_ = 0.01
        Lambda_ = 99
        # lambda_ = 0.0000000001
        # Lambda_ = 999999999
        lambda_ = 0.1
        Lambda_ = 10

        cur_condition = torch.mm(s_k_2.transpose(0, 1), y_k_2)
        cur = cur_condition

        if cur_condition > lambda_ and cur_condition < Lambda_:
            delta_D1 = torch.mm(s_k_2, s_k_2.transpose(0, 1)) / cur_condition
        else:
            print("cur not satisfy condition!")

            cur_condition = torch.mm(s_k_2.transpose(0, 1), s_k_2) * 2 / (lambda_ + Lambda_)
            # 11.29, cur_condition有可能出现0
            if cur_condition > 0:
                delta_D1 = torch.mm(s_k_2, s_k_2.transpose(0, 1)) / cur_condition
            else:
                delta_D1 = 0

            # # 11.30
            # if cur_condition > 0 and cur_condition < 1:
            #     delta_D1 = torch.mm(s_k_2, s_k_2.transpose(0, 1)) / cur_condition
            # else:
            #     delta_D1 = 0

        msg = "epoch: " + str(epoch) + " cur = " + str(cur) + " cur_condition = " + str(cur_condition)
        print(msg)

        div_2 = torch.mm(torch.mm(y_k_2.transpose(0, 1), self.D_k_2), y_k_2)
        if div_2 == 0:
            delta_D2 = 0
            print("delta_D2 = 0: torch.mm(torch.mm(y_k_2.transpose(0, 1), self.D_k_2), y_k_2) == 0")
        else:
            # delta_D2 = self.D_k_2 * y_k_2 * y_k_2.transpose(0, 1) * self.D_k_2 / torch.mm(torch.mm(y_k_2.transpose(0, 1), self.D_k_2), y_k_2)
            delta_D2 = torch.mm(torch.mm(self.D_k_2, y_k_2), torch.mm(y_k_2.transpose(0, 1), self.D_k_2)) / div_2

        delta_D = delta_D1 - delta_D2

        # # 11.30
        # if cur > Lambda_:
        #     delta_D = 0

        del delta_D1
        del delta_D2
        # D_k_1 = self.D_k_2 + delta_D
        D_k_1 = torch.add(self.D_k_2, delta_D)

        del delta_D

        self.x_k_1 = self.x_k_1.reshape(self.model_length, 1)
        self.x_k = self.x_k.reshape(self.model_length, 1)

        # eta = 1
        # eta = 0.07
        eta = 0.1
        # if self.from_layer == 0:
        #     eta = 0.1
        # eta = 0.7

        self.x_k = (self.x_k_1 - eta * torch.mm(D_k_1, self.x_k_1) / self.learning_rate) + (
                eta * torch.mm(D_k_1, self.x_k) / self.learning_rate)

        self.D_k_2 = D_k_1
        x_k = self.transfer_matrix_to_model(self.x_k.reshape(self.model_length, ))
        return x_k

    def get_model(self):
        pass


class Optimizer_adam(Optimizer_SSO):
    def __init__(self, model, from_layer, to_layer, model_length, select_layers, device, learning_rate, config=None):
        super(Optimizer_adam, self).__init__(model, from_layer, to_layer, model_length, select_layers, device,
                                             learning_rate)

        # self.grad = copy.deepcopy(model)
        assert config != None
        self.adam = torch.optim.Adam(self.model.parameters(), lr=config.server_learning_rate)

    def update_model(self, epoch, x_k_1, x_k):
        self.adam.zero_grad()
        # 1. set grad, x_k_1
        for model_p, x_k_p, x_k_1_p in zip(self.model.parameters(), x_k.parameters(), x_k_1.parameters()):
            model_p.data = x_k_1_p.data.clone()
            # model_p.grad = (x_k_1_p.data.clone() - x_k_p.data.clone()) / self.learning_rate
            model_p.grad = x_k_1_p.data.clone() - x_k_p.data.clone()

        def closure():
            return 0.0

        self.adam.step(closure)

        # list_param = []
        # for name, param in self.model.named_parameters():
        #     if name in self.select_layers:
        #         list_param.extend(list(param.data.clone()))
        #
        # return list_param
        return list(self.model.parameters())[self.from_layer: self.to_layer]

        # eta = 1 + self.learning_rate
        # eta = self.learning_rate
        # self.x_k_1 = self.transfer_model_to_matrix(x_k_1).reshape(self.model_length, 1)
        # self.x_k = self.transfer_model_to_matrix(x_k).reshape(self.model_length, 1)
        # self.g_k_1 = (self.x_k_1 - self.x_k) / (self.learning_rate)
        # self.x_k = self.x_k_1 - eta * self.g_k_1
        # x_k = self.transfer_matrix_to_model(self.x_k.reshape(self.model_length, ))
        # return x_k

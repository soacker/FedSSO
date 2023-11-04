from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_data, read_client_data
import torch
import copy

'''
直接修改了 FedBLH，添加了 lambda 和 Lambda 的界，限定 B，测试
'''


class ServerSSO_enforce3_2023(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps,
                 num_clients,
                 total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal,
                 time_threthold, eta=0.07, config=None):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select,
                         goal,
                         time_threthold)
        # select slow clients
        self.set_slow_clients()

        self.eta = eta

        assert config != None
        self.config = config

        for i, train_slow, send_slow in zip(range(self.total_clients), self.train_slow_clients, self.send_slow_clients):
            train, test = read_client_data(dataset, i)
            client = clientAVG(device, i, train_slow, send_slow, train,
                               test, model, batch_size, learning_rate, local_steps)
            self.clients.append(client)

        print(
            f"Number of clients / total clients: {self.num_clients} / {self.total_clients}")
        print("Finished creating server and clients.")

        self.global_gradient = None
        self.global_pre_gradient = None
        self.pre_pre_model = None

        self.g_k_2 = 0
        self.g_k_1 = 0
        self.x_k_2 = 0
        self.x_k_1 = 0
        self.x_k = 0
        self.D_k_2 = 0

        self.model_view_0 = 0
        self.model_view_1 = 0

        self.m = []

        self.model_length = 0

        self.model_view = []

        self.select_layers = []

        self.device = device

    def train(self):
        for i in range(self.global_rounds + 1):
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.my_aggregate_parameters(i)

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()

    # def transfer_deepmodel_to_matrix(self, model):
    #     param_list = []
    #     for param in model.parameters():
    #         if len(param.data.size()) == 2:
    #             x1 = param.resize(param.data.size()[0] * param.data.size()[1], )
    #         if len(param.data.size()) == 1:
    #             x1 = param
    #         if len(param.data.size()) == 3:
    #             x1 = param.resize(param.data.size()[0] * param.data.size()[1] * param.data.size()[2], )
    #         if len(param.data.size()) == 4:
    #             x1 = param.resize(param.data.size()[0] * param.data.size()[1] * param.data.size()[2] * param.data.size()[3], )
    #         if len(param.data.size()) == 5:
    #             x1 = param.resize(param.data.size()[0] * param.data.size()[1] * param.data.size()[2] * param.data.size()[3] * param.data.size()[4], )
    #         if len(param.data.size()) == 6:
    #             x1 = param.resize(param.data.size()[0] * param.data.size()[1] * param.data.size()[2] * param.data.size()[3] * param.data.size()[4] * param.data.size()[5], )
    #         param_list.append(x1)
    #
    #     if len(param_list) == 2:
    #         x = torch.cat((param_list[0], param_list[1]), 0)
    #     if len(param_list) == 4:
    #         x = torch.cat((param_list[0], param_list[1], param_list[2], param_list[3]), 0)
    #     if len(param_list) == 6:
    #         x = torch.cat((param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], param_list[5]), 0)
    #
    #     # x = torch.cat((param_list[0], param_list[1], param_list[2], param_list[3]), 0)
    #     return x
    #
    # def transfer_matrix_to_deepmodel(self, matrix):
    #     m = []
    #     bg = 0
    #     id = 0
    #     for ed in self.dim_lengths:
    #         x1 = matrix[bg: ed].reshape(self.model_view[id])
    #         id += 1
    #         bg = ed
    #         m.append(x1)
    #     return m
    #
    # def transfer_model_to_matrix(self, model):
    #
    #     if self.is_deep == 0:
    #         return self.transfer_deepmodel_to_matrix(model)
    #
    #     params = model.parameters()
    #
    #     for param, i in zip(params, [0, 1]):
    #         if i == 0:
    #             x1 = param.resize(param.size()[0] * param.size()[1], )
    #             self.model_view_0 = param.size()
    #         else:
    #             x2 = param
    #             self.model_view_1 = param.size()
    #
    #     x = torch.cat((x1, x2), 0)
    #     return x
    #
    #
    # def transfer_matrix_to_model(self, matrix):
    #
    #     if self.is_deep == 0:
    #         return self.transfer_matrix_to_deepmodel(matrix)
    #
    #     m = []
    #     m.append(matrix[0: self.model_view_0[0] * self.model_view_0[1]].reshape(self.model_view_0[0], self.model_view_0[1]))
    #     m.append(matrix[self.model_view_0[0] * self.model_view_0[1]: ].reshape(self.model_view_1[0],))
    #     return m

    @staticmethod
    def parameter_length(size: torch.Size):
        if len(size) >= 5:
            raise TypeError
        l = torch.prod(torch.tensor(size)).item()
        if l != int(l):
            raise TypeError
        return int(l)

    # 将所选择层参数展开成一维并连接
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
        param_dict = self.global_model.state_dict()
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

    @torch.no_grad()
    def my_aggregate_parameters(self, epoch):
        assert (len(self.uploaded_models) > 0)

        pre_model = copy.deepcopy(self.global_model)
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        # if epoch == 0:
        #     # 进行初始化
        #     self.pre_pre_model = copy.deepcopy(self.global_model)
        #     self.global_gradient = 0
        #
        #     # self.model_view = []
        #     # self.dim_lengths = []
        #     temp_length = 0
        #     self.model_length = 0
        #     self.dim_lengths = []
        #     self.model_view = []
        #
        #     for param in self.global_model.parameters():
        #
        #         if len(param.data.size()) == 2:
        #             self.model_length += param.data.size()[0] * param.data.size()[1]
        #         if len(param.data.size()) == 1:
        #             self.model_length += param.data.size()[0]
        #         if len(param.data.size()) == 3:
        #             self.model_length += param.data.size()[0] * param.data.size()[1] * param.data.size()[2]
        #         if len(param.data.size()) == 4:
        #             self.model_length += param.data.size()[0] * param.data.size()[1] * param.data.size()[2]* param.data.size()[3]
        #
        #         self.dim_lengths.append(self.model_length)
        #         self.model_view.append(param.size())
        if epoch == 0:
            # 进行初始化
            self.model_length = 0
            # self.dim_lengths = []
            # self.model_view = []

            self.pre_pre_model = copy.deepcopy(self.global_model)
            # self.global_gradient = 0

            for layer, param in self.global_model.state_dict().items():
                param_len = self.parameter_length(param.data.size())
                if param_len < 1000000 and param_len != 1:
                    self.select_layers.append(layer)
                    self.model_length += param_len
                    print('layer {} len {} selected'.format(layer, param_len))
                else:
                    print('layer {} len {} dropped'.format(layer, param_len))

            self.model_length = int(self.model_length)
            # assert (self.model_length ** 2 <= 1024 * 1024 * 1024 * 4)  # (len, len) matrix size not too huge
            print("total model selected param len {}, select_layers {}".format(
                self.model_length, self.select_layers))

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        if epoch == 0:
            self.x_k_1 = 0
            self.g_k_1 = 0
            self.D_k_2 = torch.eye(self.model_length).to(self.device)

        # 根据DFP
        # 替换

        # 深度模型，此处赋值0
        self.is_deep = 0

        self.x_k_2 = self.x_k_1
        self.x_k_1 = self.transfer_model_to_matrix(pre_model).reshape(self.model_length, 1)
        self.x_k = self.transfer_model_to_matrix(self.global_model).reshape(self.model_length, 1)

        del pre_model

        self.g_k_2 = self.g_k_1
        self.g_k_1 = (self.x_k_1 - self.x_k) / (self.learning_rate)  # * self.local_steps

        sk_sigma = 0.7
        s_k_2 = (self.x_k_1 - self.x_k_2).reshape(self.model_length, 1) * sk_sigma  #
        y_k_2 = (self.g_k_1 - self.g_k_2).reshape(self.model_length, 1)

        # lambda_ = 0.0000000001
        # Lambda_ = 999999999
        min_lambda_ = self.config.min_lambda
        max_lambda_: float = self.config.max_lambda

        # torch.dot(s_k_2.reshape(7850, ), y_k_2.reshape(7850, ))
        # div_1 = torch.mm(s_k_2.transpose(0, 1), y_k_2)

        cur_condition = torch.mm(s_k_2.transpose(0, 1), y_k_2)
        cur = cur_condition

        if min_lambda_ < cur_condition < max_lambda_:
            delta_D1 = torch.mm(s_k_2, s_k_2.transpose(0, 1)) / cur_condition
        else:
            print("cur not satisfy condition!")

            cur_condition = torch.mm(s_k_2.transpose(0, 1), s_k_2) * 2 / (min_lambda_ + max_lambda_)
            # 11.29, cur_condition有可能出现0
            if cur_condition > 0:
                delta_D1 = torch.mm(s_k_2, s_k_2.transpose(0, 1)) / cur_condition
            else:
                delta_D1 = 0
        msg = "epoch: " + str(epoch) + " cur = " + str(cur) + " cur_condition = " + str(cur_condition)
        print(msg)

        # # if torch.dot(s_k_2.reshape(7850, ), y_k_2.reshape(7850, )) == 0:
        # if div_1 == 0:
        #     # torch.mm(s_k_2.transpose(0, 1), y_k_2) == 0
        #     print("delta_D1 = 0: torch.dot(s_k_2.reshape(7850, ), y_k_2.reshape(7850, )) == 0")
        #     delta_D1 = 0
        # else:
        #     # delta_D1 = s_k_2 * s_k_2.transpose(0, 1) / torch.dot(s_k_2.reshape(7850, ), y_k_2.reshape(7850, ))
        #     delta_D1 = torch.mm(s_k_2, s_k_2.transpose(0, 1)) / div_1

        # if y_k_2.transpose(0, 1) * self.D_k_2 * y_k_2 == 0:

        div_2 = torch.mm(torch.mm(y_k_2.transpose(0, 1), self.D_k_2), y_k_2)
        if div_2 == 0:
            delta_D2 = 0
            print("delta_D2 = 0: torch.mm(torch.mm(y_k_2.transpose(0, 1), self.D_k_2), y_k_2) == 0")
        else:
            # delta_D2 = self.D_k_2 * y_k_2 * y_k_2.transpose(0, 1) * self.D_k_2 / torch.mm(torch.mm(y_k_2.transpose(0, 1), self.D_k_2), y_k_2)
            delta_D2 = torch.mm(torch.mm(self.D_k_2, y_k_2), torch.mm(y_k_2.transpose(0, 1), self.D_k_2)) / div_2

        delta_D = delta_D1 - delta_D2

        del delta_D1
        del delta_D2
        # D_k_1 = self.D_k_2 + delta_D
        D_k_1 = torch.add(self.D_k_2, delta_D)

        del delta_D

        self.x_k_1 = self.x_k_1.reshape(self.model_length, 1)
        self.x_k = self.x_k.reshape(self.model_length, 1)

        # sigma = 1  # 0.3 1 # 0.3
        #
        # beta = 0.1
        # # 12-21
        # beta = 0
        # self.x_k = (self.x_k_1 - sigma * torch.mm(D_k_1, self.x_k_1) / self.learning_rate) / (1 - beta ** (epoch + 1)) + sigma * torch.mm(D_k_1, self.x_k) / self.learning_rate

        # eta = 1 #3 # 2 # 0.7 # 0.7 # 0.1
        # eta = self.eta
        # eta = 1
        eta = self.config.server_learning_rate

        # step = eta / (self.learning_rate ) # * self.local_steps
        # self.x_k = self.x_k_1 - step * torch.mm(D_k_1, self.x_k_1) + step * torch.mm(D_k_1, self.x_k)
        self.x_k = (self.x_k_1 - eta * torch.mm(D_k_1, self.x_k_1) / self.learning_rate) + (
                    eta * torch.mm(D_k_1, self.x_k) / self.learning_rate)

        self.D_k_2 = D_k_1

        x_k = self.transfer_matrix_to_model(self.x_k.reshape(self.model_length, ))

        for server_param, cu_param in zip(self.global_model.parameters(), x_k):
            server_param.data = cu_param.data.clone()
            # server_param.data = cu_param.data

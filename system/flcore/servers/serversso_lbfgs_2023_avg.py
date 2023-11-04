import time

from flcore.clients.clientavg import clientAVG
from flcore.optimizers.sso_optimizer import sso_optimizer
from flcore.servers.serverbase import Server
from utils.data_utils import read_data, read_client_data
from threading import Thread
import copy


class FedSSO_sso_optimizer_2023_avg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                 total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold,
                 config=None):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        # select slow clients
        self.x_k_1 = None
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

        assert config != None
        self.sso_optimizer = sso_optimizer(self.global_model.parameters(), lr=config.server_learning_rate, history_size=global_rounds,
                                           local_steps=local_steps, lamb_=config.min_lambda, Lamb_=config.max_lambda,
                                           all=0, enforce=2)

    def train(self):
        for i in range(self.global_rounds+1):

            s_t = time.time()
            x_k_1 = copy.deepcopy(self.global_model)
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

            x_k = copy.deepcopy(self.global_model)

            self.sso_optimizer.zero_grad()
            for model_p, x_k_p, x_k_1_p in zip(self.global_model.parameters(), x_k.parameters(), x_k_1.parameters()):
                model_p.data = x_k_1_p.data.clone()
                model_p.grad = (x_k_1_p.data.clone() - x_k_p.data.clone()) / self.learning_rate
                # model_p.grad = x_k_1_p.data.clone() - x_k_p.data.clone()

            self.sso_optimizer.step()

            print('-' * 25, 'client trains time cost', '-' * 25, time.time() - s_t)
            if self.stop_sigma:
                break


        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()

import argparse
import os
import time
import warnings
from flcore.servers.ServerSSO_enforce3_2023 import ServerSSO_enforce3_2023
from flcore.servers.server_adagrad_paper_2023 import server_adagrad_paper_2023
from flcore.servers.server_adam_paper_2023 import server_adam_paper_2023
from flcore.servers.server_yogi_paper_2023 import server_yogi_paper_2023
from flcore.servers.serveravg import FedAvg
from flcore.servers.serversso_lbfgs_2023_avg import FedSSO_sso_optimizer_2023_avg
from flcore.trainmodel.models import *
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")

import torch
import numpy as np
import random


def seed_function(seed=0):
    # seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 当有多张GPU时，为所有GPU设置种子
    np.random.seed(seed)  # 为Numpy设置随机种子
    random.seed(seed)  #


def run(goal, dataset, num_labels, device, algorithm, model, local_batch_size, local_learning_rate, global_rounds,
        local_steps, num_clients,
        total_clients, beta, lamda, K, personalized_learning_rate, times, eval_gap, client_drop_rate, train_slow_rate,
        send_slow_rate,
        time_select, time_threthold, M, mu, itk, alphaK, sigma, xi, config=None):
    time_list = []
    reporter = MemReporter()

    for i in range(times):

        seed_function(i * 100)

        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        server, Head, Model, Classifier = None, None, None, None

        if model == "mclr":
            if "mnist" in dataset:
                Model = Mclr_Logistic(1 * 28 * 28, num_labels=num_labels).to(device)
            elif "Cifar" in dataset:
                Model = Mclr_Logistic(3 * 32 * 32, num_labels=num_labels).to(device)
            else:
                raise NotImplementedError("dataset don't contain")
        elif model == "LeNET":
            if "Cifar10" in dataset:
                Model = Cifar10LeNetSmall(num_labels=num_labels).to(device)
            else:
                raise NotImplementedError("no finish")
        elif model == "resnet20":
            if "Cifar100" in dataset:
                # Model =
                from flcore.trainmodel.print_model_ import resnet20
                Model = resnet20()

                Model.linear = torch.nn.Linear(64, 100)

                # Model.linear = torch.nn.Sequential(
                #     torch.nn.Linear(64, 100),
                #     torch.nn.ReLU(True),
                #     torch.nn.Dropout(),
                #     torch.nn.Linear(100, 100)
                # )
                Model = Model.to(device)

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate,
                            send_slow_rate, time_select, goal, time_threthold)
        elif algorithm == "FedSSO-v1":
            server = ServerSSO_enforce3_2023(device, dataset, algorithm, Model, local_batch_size, local_learning_rate,
                                             global_rounds,
                                             local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate,
                                             train_slow_rate,
                                             send_slow_rate, time_select, goal, time_threthold, config=config)
        elif algorithm == "FedSSO-v2":
            server = FedSSO_sso_optimizer_2023_avg(device, dataset, algorithm, Model, local_batch_size,
                                                   local_learning_rate, global_rounds,
                                                   local_steps, num_clients, total_clients, i, eval_gap,
                                                   client_drop_rate, train_slow_rate,
                                                   send_slow_rate, time_select, goal, time_threthold, config=config)
        elif algorithm == "FedAdam":
            server = server_adam_paper_2023(device, dataset, algorithm, Model, local_batch_size, local_learning_rate,
                                            global_rounds,
                                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate,
                                            train_slow_rate,
                                            send_slow_rate, time_select, goal, time_threthold, config=config)
        elif algorithm == "FedYogi":
            server = server_yogi_paper_2023(device, dataset, algorithm, Model, local_batch_size, local_learning_rate,
                                            global_rounds,
                                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate,
                                            train_slow_rate,
                                            send_slow_rate, time_select, goal, time_threthold, config=config)
        elif algorithm == "FedAdagrad":
            server = server_adagrad_paper_2023(device, dataset, algorithm, Model, local_batch_size, local_learning_rate,
                                               global_rounds,
                                               local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate,
                                               train_slow_rate,
                                               send_slow_rate, time_select, goal, time_threthold, config=config)

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=dataset, algorithm=algorithm, goal=goal, times=times, length=global_rounds / eval_gap + 1)

    # Personalization average
    if algorithm == "pFedMe":
        average_data(dataset=dataset, algorithm=algorithm + '_p', goal=goal, times=times,
                     length=global_rounds / eval_gap + 1)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="lambda-0.1",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str,
                        default="Cifar10-20-dir-1")
    parser.add_argument('-nb', "--num_labels", type=int, default=10)  # shakespeare 类别 65
    parser.add_argument('-niid', "--noniid", type=bool, default=True)
    parser.add_argument('-m', "--model", type=str,
                        default="LeNET")
    parser.add_argument('-lbs', "--local_batch_size", type=int, default=100)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,  # 0.03  0.1 0.001  0.0001
                        help="Local learning rate")
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1,  # 0.03  0.1 0.001  0.0001
                        help="Server learning rate")
    parser.add_argument('-minlambda', "--min_lambda", type=float, default=0.00001,  # 0.03  0.1 0.001  0.0001
                        help="min lambda")
    parser.add_argument('-maxlambda', "--max_lambda", type=float, default=100000,  # 0.03  0.1 0.001  0.0001
                        help="max lambda")
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_steps", type=int, default=10)
    parser.add_argument('-algo', "--algorithm", type=str,
                        default="FedAdagrad-paper-2023")  # FedSSO-sso-optimizer3-avg FedBLH FedSSO_enforce2 FedAvg_fn_mycnn
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Number of clients per round")
    parser.add_argument('-tc', "--total_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=5,
                        help="Running times")
    parser.add_argument('-sd', "--seed", type=int, default=1,
                        help="Running seed")

    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / HeurFedAMP
    parser.add_argument('-bt', "--beta", type=float, default=0.001,  # 0.0
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg")
    parser.add_argument('-lam', "--lamda", type=float, default=15,  # 15
                        help="Regularization weight for pFedMe and FedAMP")
    # FedAC
    parser.add_argument('-mu', "--mu", type=float, default=0.001,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--personalized_learning_rate", type=float, default=0.001,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=3,
                        help="Server only sends M client models to one client at each round")
    # MOCHA
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # HeurFedAMP
    parser.add_argument('-xi', "--xi", type=float, default=1.0)

    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

    if config.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        config.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(config.algorithm))
    print("Local batch size: {}".format(config.local_batch_size))
    print("Local steps: {}".format(config.local_steps))
    print("Local learing rate: {}".format(config.local_learning_rate))
    print("Total clients: {}".format(config.total_clients))
    print("Client drop rate: {}".format(config.client_drop_rate))
    print("Time select: {}".format(config.time_select))
    print("Time threthold: {}".format(config.time_threthold))
    print("Subset of clients: {}".format(config.num_clients))
    print("Global rounds: {}".format(config.global_rounds))
    print("Running times: {}".format(config.times))
    print("Dataset: {}".format(config.dataset))
    print("Local model: {}".format(config.model))
    print("Using device: {}".format(config.device))

    if config.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    elif config.algorithm == "pFedMe":
        print("Average moving parameter beta: {}".format(config.beta))
        print("Regularization rate: {}".format(config.lamda))
        print("Number of personalized training steps: {}".format(config.K))
        print("personalized learning rate to caculate theta: {}".format(config.personalized_learning_rate))
    elif config.algorithm == "PerAvg":
        print("Second learning rate beta: {}".format(config.beta))
    elif config.algorithm == "FedProx":
        print("Proximal rate: {}".format(config.mu))
    elif config.algorithm == "FedFomo":
        print("Server sends {} models to one client at each round".format(config.M))
    elif config.algorithm == "MOCHA":
        print("The iterations for solving quadratic subproblems: {}".format(config.itk))
    elif config.algorithm == "FedAMP":
        print("alphaK: {}".format(config.alphaK))
        print("lamda: {}".format(config.lamda))
        print("sigma: {}".format(config.sigma))
    elif config.algorithm == "HeurFedAMP":
        print("alphaK: {}".format(config.alphaK))
        print("lamda: {}".format(config.lamda))
        print("sigma: {}".format(config.sigma))
        print("xi: {}".format(config.xi))

    print("=" * 50)

    run(
        goal=config.goal,
        dataset=config.dataset,
        num_labels=config.num_labels,
        device=config.device,
        algorithm=config.algorithm,
        model=config.model,
        local_batch_size=config.local_batch_size,
        local_learning_rate=config.local_learning_rate,
        global_rounds=config.global_rounds,
        local_steps=config.local_steps,
        num_clients=config.num_clients,
        total_clients=config.total_clients,
        beta=config.beta,
        lamda=config.lamda,
        K=config.K,
        personalized_learning_rate=config.personalized_learning_rate,
        times=config.times,
        eval_gap=config.eval_gap,
        client_drop_rate=config.client_drop_rate,
        train_slow_rate=config.train_slow_rate,
        send_slow_rate=config.send_slow_rate,
        time_select=config.time_select,
        time_threthold=config.time_threthold,
        M=config.M,
        mu=config.mu,
        itk=config.itk,
        alphaK=config.alphaK,
        sigma=config.sigma,
        xi=config.xi,
        config=config,
    )

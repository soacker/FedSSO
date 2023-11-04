import csv

import datetime

import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, length=800):
    test_acc, train_acc, train_loss = np.array([[0]]), np.array([[0]]), np.array([[0]])
    try:
        test_acc, train_acc, train_loss = get_all_results_for_one_algo(
            algorithm, dataset, goal, times, int(length))
    except:
        print("Nan")
    test_acc_data = np.average(test_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))
    std = np.std(max_accurancy)
    mean = np.mean(max_accurancy)


    for t in range(times):
        file_name = dataset + "_" + algorithm + "_" + goal + "_" + str(t)
        file_path = "../results/" + file_name + ".h5"
        if (len(test_acc) != 0 & len(train_acc) & len(train_loss)):
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=test_acc[t])
                hf.create_dataset('rs_train_acc', data=train_acc[t])
                hf.create_dataset('rs_train_loss', data=train_loss[t])
        write_csv(file_name+".h5", test_acc[t].max(), train_loss[t].min(), std=std, mean=mean)


def write_csv(filename, max_test_acc, min_train_loss, std=0, mean=0):
    order = ["dataset", "algo", "m", "gr", "lbs", "ls", "lr", "slr", "nc", "cdr", "mu", "set", "date", "other", "max_test_acc", "min_train_loss", "std_for_best_accuracy", "mean_for_best_accurancy", "filename"]

    names = filename.split("_")
    record_ = dict()
    for id, n_ in enumerate(names):
        if id == 0:
            record_["dataset"] = n_
        elif id == 1:
            record_["algo"] = n_
        elif "avg" in n_ or "-" not in n_:
            continue
        else:
            record_[n_.split("-")[0]] = n_.split("-")[1]
    record_["max_test_acc"] = max_test_acc
    record_["min_train_loss"] = min_train_loss
    record_["filename"] = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + "_" + filename
    record_["std_for_best_accuracy"] = std
    record_["mean_for_best_accurancy"] = mean


    path = "record_results_gpu1.csv"
    if not os.path.exists(path):
        with open(path, "w") as f:
            csv_ = csv.writer(f)
            csv_head = order
            csv_.writerow(csv_head)
    with open(path, "a+") as f:
        csv_ = csv.writer(f)
        data_row = []
        for term in order:
            data_row.append(record_.get(term, "-"))
        csv_.writerow(data_row)







def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, length=800):
    train_acc = np.zeros((times, length))
    train_loss = np.zeros((times, length))
    test_acc = np.zeros((times, length))
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + \
            algorithms_list[i] + "_" + goal + "_" + str(i)
        # train_acc[i, :], train_loss[i, :], test_acc[i, :] = np.array(
        #     read_data_then_delete(file_name, delete=True))[:, :length]
        train_acc[i, :], train_loss[i, :], test_acc[i, :] = np.array(
            read_data_then_delete(file_name, delete=True))

    return test_acc, train_acc, train_loss


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"
    print("File path: " + file_path)

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))
        rs_train_acc = np.array(hf.get('rs_train_acc'))
        rs_train_loss = np.array(hf.get('rs_train_loss'))

    if delete:
        os.remove(file_path)

    return rs_train_acc, rs_train_loss, rs_test_acc
import multiprocessing
import subprocess


class Params(object):
    # def init_setting(self):
    # ls = [10, 50, 100]
    ls = [10, 3]
    # lr = [0.001, 0.003, 0.007, 0.01, 0.03, 0.07, 0.1, 0.3, 0.7]
    lr = [0.1]
    # slr = [0.001, 0.01, 0.1, 1]
    slr = [0.01]
    m = ["LeNET"]
    data = ["Cifar10-20-dir-1"]
    algo = ["FedSSO_enforce3_2023"]
    tc = [20]
    nc = [20]
    lbs = [100]
    gr = [1]
    # cdr = [0, 0.1, 0.2, 0.3]
    cdr = [0]
    nb = [10]

    minlambda = [0.0000001]
    maxlambda = [10000000]

def init_params():
    example = Params()
    members = [attr for attr in dir(example) if not callable(getattr(example, attr)) and not attr.startswith("__")]
    print(members)
    params = {}
    for member in members:
        params.update({member: example.__getattribute__(member)})
    print(params)
    return params

def deep_search(i, res, record):
    if i == len(keys):
        global_params.append(res)
        global_records.append(record)
        return
    key = keys[i]
    value = params[key]
    for v in value:
        temp_res = res
        temp_record = record
        res += "-" + key + " " + str(v) + " "
        record += key + "-" + str(v) + "_"
        deep_search(i + 1, res, record)
        res = temp_res
        record = temp_record

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr

def run_pool():
    pool = multiprocessing.Pool(processes=20)

    # 使用进程池执行命令
    results = []
    num = 0
    for command, goal in zip(global_params, global_records):
        cmd = "/data/persist/.conda/envs/py310/bin/python system/main_my.py " + "-did " + str(num % 2) + " " + command + "-go " + goal
        print(cmd)
        result = pool.apply_async(run_command, (cmd,))
        results.append(result)
        num += 1

    # 等待所有进程完成
    pool.close()
    pool.join()

    # 打印命令的执行结果
    for i, result in enumerate(results):
        stdout, stderr = result.get()
        print(f"Command {i + 1} stdout:\n{stdout}")
        print(f"Command {i + 1} stderr:\n{stderr}")


params = init_params()
keys = list(params.keys())
global_params = []
global_records = []


deep_search(0, "", "")
# print(global_params)
# print(global_records)

run_pool()



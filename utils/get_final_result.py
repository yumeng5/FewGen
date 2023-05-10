import argparse
import numpy as np
import os

task_metrics = {
    "mnli": "acc",
    "sst-2": "acc",
    "rte": "acc",
    "qnli": "acc",
    "cola": "mcc",
    "qqp": "f1",
    "mrpc": "f1",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='final_result/')
args = parser.parse_args()
print(args)
for _, subdirs, _ in os.walk(args.dir):
    break


for subdir in subdirs:
    print(f'\ntask: {subdir}')
    metric = task_metrics[subdir.lower()]
    dir_name = os.path.join(args.dir, subdir)
    for _, seed_dirs, files in os.walk(dir_name):
        break
    all_res = []
    all_res_ = []
    for seed_dir in seed_dirs:
        seed_dir_name = os.path.join(args.dir, subdir, seed_dir)
        if not os.path.exists(os.path.join(seed_dir_name, f"test_results_{subdir.lower()}.txt")):
            print(f"test file not exist for {seed_dir_name}!")
            # exit()
            continue
        else:
            if subdir != "MNLI":
                f = open(os.path.join(seed_dir_name, f"test_results_{subdir.lower()}.txt"))
                res = f.readlines()
                for line in res:
                    title, val = line.strip().split(' = ')
                    if title == f"eval_{metric}":
                        print(f"seed {seed_dir}\t {float(val)}")
                        all_res.append(float(val))
            else:
                f = open(os.path.join(seed_dir_name, f"test_results_{subdir.lower()}.txt"))
                res = f.readlines()
                for line in res:
                    title, val = line.strip().split(' = ')
                    if title == f"eval_mnli/{metric}":
                        print(f"seed {seed_dir}\t {float(val)}")
                        all_res.append(float(val))
                f = open(os.path.join(seed_dir_name, f"test_results_{subdir.lower()}-mm.txt"))
                res = f.readlines()
                for line in res:
                    title, val = line.strip().split(' = ')
                    if title == f"eval_mnli-mm/{metric}":
                        print(f"seed {seed_dir}\t {float(val)}")
                        all_res_.append(float(val))
    print(all_res)
    print(f"mean: {100*np.average(all_res)},  std: {100*np.std(all_res)}")
    if len(all_res_) > 0:
        print(all_res_)
        print(f"mean: {100*np.average(all_res_)},  std: {100*np.std(all_res_)}")

import json
import numpy as np
import argparse
import os
from collections import defaultdict


task_label = {
    "mnli": ["entailment", "neutral", "contradiction"],
    "qqp": ["0", "1"],
    "qnli": ["entailment", "not_entailment"],
    "sst-2": ["0", "1"],
    "cola": ["0", "1"],
    "rte": ["entailment", "not_entailment"],
    "mrpc": ["0", "1"],
}


def read_files(read_dir, task):
    for (_, _, filenames) in os.walk(read_dir):
        break
    file_dict = {}
    for label in task_label[task]:
        found = False
        for f in filenames:
            if f.startswith(f"{task}_{label}") and f.endswith(".json"):
                if found:
                    print(f"Found more than one generated file for task {task}, label {label}! Make sure there is only one!")
                    exit(-1)
                found = True
                file_dict[label] = os.path.join(read_dir, f)
        if not found:
            print(f"Not found generated file for task {task}, label {label}!")
            exit(-1)
    return file_dict


def combine(gen_file_dict, k=None):
    combined_dict = []
    data_dicts = []
    for label, file_dir in gen_file_dict.items():
        data_dict = json.load(open(file_dir, 'r'))
        print(f"Label {label}: {len(data_dict)} total samples")
        data_dicts.append(data_dict)
    if k is None:
        k = max([len(data_dict) for data_dict in data_dicts])
    label_count = defaultdict(int)
    for i in range(k):
        for data_dict in data_dicts:
            if i < len(data_dict):
                combined_dict.append(data_dict[i])
                label_count[data_dict[i]["label"]] += 1
    for label in label_count:
        print(f"Label {label}: {label_count[label]} selected samples")
    print(f"Total {len(combined_dict)} samples")
    return combined_dict


def save(save_file_dir, save_dict):
    with open(save_file_dir, 'w') as f:
        res = json.dumps(save_dict)
        f.write(res)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--seed',)
    args = parser.parse_args()
    task = args.task.lower()
    read_dir = f"gen_res_{task}_{args.seed}"
    save_dir = f"data/k-shot/{args.task}/16-{args.seed}/"
    gen_file_dict = read_files(read_dir, task)
    combined_dict = combine(gen_file_dict)
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f"gen-train.json")
    save(save_name, combined_dict)
    print(f"Combined training set saved to {save_name}")

if __name__ == "__main__":
    main()
    
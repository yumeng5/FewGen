from genericpath import exists
import os
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--dir',)
args = parser.parse_args()
print(args)
for _, subdirs, _ in os.walk(args.dir):
    break

best_eval_res = -100
best_eval_loss = 100
best_param = None
for subdir in sorted(subdirs):
    if subdir == 'best':
        dir_name = os.path.join(args.dir, subdir)
        for _, _, files in os.walk(dir_name):
            break
        for file in sorted(files):
            if file.startswith("best_param.txt"):
                f = open(os.path.join(dir_name, file))
                contents = f.readlines()
                best_dir = contents[0].strip()
                res = float(contents[1].strip())
    else:    
        dir_name = os.path.join(args.dir, subdir)
        for _, _, files in os.walk(dir_name):
            break
        for file in files:
            if file.startswith("eval_results"):
                f = open(os.path.join(dir_name, file))
                contents = f.readlines()
                loss = float(contents[0].strip().split(' = ')[1])
                res = float(contents[1].strip().split(' = ')[1])
        try:
            if res > best_eval_res:
                best_eval_res = res
                best_eval_loss = loss
                best_param = subdir
        except:
            print(f"{dir_name} training failed!")

save_dir = os.path.join(args.dir, "best")
os.makedirs(save_dir, exist_ok=True)
f = open(os.path.join(save_dir, "best_param.txt"), 'w')
f.write(best_param + '\n' + str(best_eval_res) + '\n')
print(f"Best eval res: {best_eval_res}")
print(f"Best parameter: {best_param}")
print(f"Copying best checkpoint to {save_dir}")
try:
    shutil.copyfile(os.path.join(args.dir, best_param, "pytorch_model.bin"), os.path.join(save_dir, "pytorch_model.bin"))
    shutil.copyfile(os.path.join(args.dir, best_param, "config.json"), os.path.join(save_dir, "config.json"))

    # remove all checkpoints not selected
    print(f"Removing other checkpoints")
    for subdir in subdirs:
        dir_name = os.path.join(args.dir, subdir)
        if 'best' not in subdir and os.path.exists(os.path.join(dir_name, "pytorch_model.bin")):
            os.remove(os.path.join(dir_name, "pytorch_model.bin"))
except:
    print(f"Best checkpoint cannot be found!")
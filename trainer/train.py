import os
import json
import shutil
import importlib
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.multiprocessing as mp

from trainer.worker_rtgs import worker
from trainer.learner import learner
from trainer.evaluator import evaluator
from config import BASEDIR


def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"] + "/args.json", "w")
    f.write(args_info)
    f.close()


def copy_models(dir_src, dir_dst):
    # retireve list of models
    l_cands = [f for f in os.listdir(dir_src) if os.path.isfile(os.path.join(dir_src, f)) and 'model_' in f]
    l_cands = sorted(l_cands, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"models to be copied: {l_cands}")
    for m in l_cands:
        shutil.copyfile(os.path.join(dir_src, m), os.path.join(dir_dst, m))
    print(f"{len(l_cands)} models copied in the given directory")


def main(arg_dict):
    cpu_num = 1  # set CPU number
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    cur_time = datetime.now() + timedelta(hours=0)
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    save_args(arg_dict)
    if arg_dict["trained_model_path"]:
        copy_models(os.path.dirname(arg_dict['trained_model_path']), arg_dict['log_dir'])

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    imported_encoder = importlib.import_module("encoders." + arg_dict["encoder"])
    fe = imported_encoder.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()

    imported_model = importlib.import_module("models." + arg_dict["model"])
    cpu_device = torch.device('cpu')
    center_model = imported_model.Model(arg_dict)

    if arg_dict["trained_model_path"]:
        checkpoint = torch.load(arg_dict["trained_model_path"], map_location=cpu_device)
        optimization_step = checkpoint['optimization_step']
        center_model.load_state_dict(checkpoint['model_state_dict'])
        center_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        arg_dict["optimization_step"] = optimization_step
        print("Trained model", arg_dict["trained_model_path"], "successful loaded")
    else:
        optimization_step = 0

    model_dict = {
        'optimization_step': optimization_step,
        'model_state_dict': center_model.state_dict(),
        'optimizer_state_dict': center_model.optimizer.state_dict(),
    }

    path = arg_dict["log_dir"] + f"/model_{optimization_step}.tar"
    torch.save(model_dict, path)

    center_model.share_memory()
    data_queue = mp.Queue()
    summary_queue = mp.Queue()
    signal_queue = mp.Queue()

    processes = []
    p = mp.Process(target=learner, args=(center_model, data_queue, signal_queue, summary_queue, arg_dict))
    p.start()
    processes.append(p)
    for rank in range(arg_dict["workers"]):
        p = mp.Process(target=worker, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)

    if "env_evaluation" in arg_dict:
        p = mp.Process(target=evaluator, args=(center_model, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    arg_dict = {
        "env": "11_vs_11_stochastic",
        "workers": 25,  # if test: 1
        "batch_size": 32,  # default 32, if test: 2.
        "buffer_size": 3,  # default 3
        "rollout_len": 30,  # default 30, if test: 3.
        "trained_model_path": None,  # use when you want to continue training from given model.

        "encoder": "encoder",
        "rewarder": "rewarder",
        "model": "conv1d",
        "algorithm": "ppo_lambda",

        "lstm_size": 256,
        "k_epoch": 3,
        "learning_rate": 0.0001,  # 1e-4
        "gamma": 0.993,
        "lmbda": 0.5,
        "entropy_coef": 0.0001,  # entropy regularization
        "grad_clip": 3.0,
        "eps_clip": 0.1,

        "summary_game_window": 30,
        "model_save_interval": 20000,
        "env_evaluation": '11_vs_11_stochastic',  # 11_vs_11_hard_stochastic

        # gail
        "use_gail": False,
        "gail_epoch": 4,
        # "gali_reg_coef":  1,
        "gail_batch_size": 64,  # test 2
        "exp_data_path": BASEDIR + '/data/exp_traj.pkl',
        "gail_save_interval": 2000

    }
    main(arg_dict)

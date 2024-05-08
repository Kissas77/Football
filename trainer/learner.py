import time
import importlib
import numpy as np
import torch
from tensorboardX import SummaryWriter


def write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst,
                  gail_pi_loss_lst, gail_exp_loss_lst, gail_pi_acc_list, gail_exp_acc_list, optimization_step,
                  win_evaluation, score_evaluation):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        w, s, r, step, opp_num, t1, t2, t3 = game_data

        if 'env_evaluation' in arg_dict and opp_num == arg_dict['env_evaluation']:
            win_evaluation.append(w)
            score_evaluation.append(s)
        else:
            win.append(w)
            score.append(s)
            tot_reward.append(r)
            game_len.append(step)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)

    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)
    writer.add_scalar('train/pi_loss', np.mean(pi_loss_lst), n_game)
    writer.add_scalar('train/v_loss', np.mean(v_loss_lst), n_game)
    writer.add_scalar('train/entropy', np.mean(entropy_lst), n_game)
    writer.add_scalar('train/move_entropy', np.mean(move_entropy_lst), n_game)
    if arg_dict['use_gail']:
        writer.add_scalar('gail/pi_loss', np.mean(gail_pi_loss_lst), n_game)
        writer.add_scalar('gail/exp_loss', np.mean(gail_exp_loss_lst), n_game)
        writer.add_scalar('gail/pi_acc', np.mean(gail_pi_acc_list), n_game)
        writer.add_scalar('gail/exp_acc', np.mean(gail_exp_acc_list), n_game)

    mini_window = max(1, int(arg_dict['summary_game_window'] / 3))
    if len(win_evaluation) >= mini_window:
        writer.add_scalar('game/evaluation_win_rate', float(np.mean(win_evaluation)), n_game)
        writer.add_scalar('game/evaluation_score', float(np.mean(score_evaluation)), n_game)
        win_evaluation, score_evaluation = [], []

    return win_evaluation, score_evaluation


def save_model(model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["log_dir"] + "/model_" + str(optimization_step) + ".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step


def save_gail_model(model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["gail_save_interval"]:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["log_dir"] + "/gail_" + str(optimization_step) + ".tar"
        torch.save(model_dict, path)
        # print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step


def get_data(queue, arg_dict, model):
    # data == buffer(mini batch)
    data = []
    for i in range(arg_dict["buffer_size"]):
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]):
            rollout = queue.get()
            mini_batch_np.append(rollout)
        mini_batch = model.make_batch(mini_batch_np)
        data.append(mini_batch)
    return data


def get_exp_data(buffer, arg_dict, model):
    # data == buffer(mini batch)
    data = []
    for i in range(arg_dict['buffer_size']):
        mini_batch = buffer.sample()
        data.append(model.make_batch(mini_batch))
    return data


def learner(center_model, queue, signal_queue, summary_queue, arg_dict):
    print("[Learner] process started")
    imported_model = importlib.import_module("models." + arg_dict["model"])
    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = imported_model.Model(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    algo = imported_algo.Algo(arg_dict)
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if torch.cuda.is_available() and isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    model.to(device)

    if arg_dict['use_gail']:
        t1 = time.time()
        imported_D = importlib.import_module("models.discriminator")
        imported_gail = importlib.import_module("algos.gail")
        imported_buffer_exp = importlib.import_module("data.buffer")
        discriminator = imported_D.Discriminator(arg_dict, device)
        discriminator.to(device)
        gail = imported_gail.Algo(arg_dict)
        buffer_exp = imported_buffer_exp.BufferExp(arg_dict)
        t2 = time.time()
        print(f'[Time] GAIL init cost {t2-t1} s')

    writer = SummaryWriter(logdir=arg_dict["log_dir"])

    optimization_step = 0
    gail_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step
    gail_saved_step = gail_step
    n_game = 0

    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst, \
        gail_pi_loss_lst, gail_exp_loss_lst, gail_pi_acc_list, gail_exp_acc_list = [], [], [], [], [], [], [], [], []
    win_evaluation, score_evaluation = [], []

    while True:
        if queue.qsize() > arg_dict["batch_size"] * arg_dict["buffer_size"]:
            # save model
            last_saved_step = save_model(model, arg_dict, optimization_step, last_saved_step)
            if arg_dict['use_gail']:
                gail_saved_step = save_gail_model(discriminator, arg_dict, gail_step, gail_saved_step)

            signal_queue.put(1)

            # make data
            # policy data
            t1 = time.time()
            data = get_data(queue, arg_dict, model)  # data shape: [buffer, batchsize, rollout, trans]
            # gail data
            if arg_dict['use_gail']:
                # replace reward
                for buffer_i in range(arg_dict['buffer_size']):
                    mini_batch = data[buffer_i]
                    state, action = mini_batch[0], mini_batch[1]
                    with torch.no_grad():
                        mini_batch[4] = discriminator.predict_reward(state, action)  # reward replace
                data_exp = get_exp_data(buffer_exp, arg_dict, discriminator)
            data_t = time.time() - t1

            # training
            # update gail todo warm up gail
            t2 = time.time()
            if arg_dict['use_gail']:
                gail_pi_loss, gail_exp_loss, gail_pi_acc, gail_exp_acc = gail.train(discriminator, data, data_exp)
            # update actor-critic
            loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, data)
            optimization_step += arg_dict["buffer_size"] * arg_dict["k_epoch"]
            # lr decay
            new_lr = arg_dict["learning_rate"] * (1 - optimization_step / 2.5e5)
            model.optimizer.param_groups[0]["lr"] = new_lr
            train_t = time.time() - t2

            # print log
            print(f"[RL] step: {optimization_step}, loss: {loss:.3f}, data_q: {queue.qsize()}, summary_q: {summary_queue.qsize()}, "
                  f"make data time: {round(data_t,4)}, train time: {round(train_t,4)}")
            if arg_dict['use_gail']:
                gail_step += arg_dict["buffer_size"] * arg_dict['gail_epoch']
                print(f"[GAIL] step: {gail_step}, gail_pi_loss: {gail_pi_loss:.3f}, gail_exp_loss: {gail_exp_loss:.3f}"
                      f" gail_pi_acc: {gail_pi_acc*100:.2f}%, gail_exp_acc: {gail_exp_acc*100:.2f}%")

            # list append
            loss_lst.append(loss)
            pi_loss_lst.append(pi_loss)
            v_loss_lst.append(v_loss)
            entropy_lst.append(entropy)
            move_entropy_lst.append(move_entropy)
            if arg_dict['use_gail']:
                gail_pi_loss_lst.append(gail_pi_loss)
                gail_exp_loss_lst.append(gail_exp_loss)
                gail_pi_acc_list.append(gail_pi_acc)
                gail_exp_acc_list.append(gail_exp_acc)

            center_model.load_state_dict(model.state_dict())

            if queue.qsize() > arg_dict["batch_size"] * arg_dict["buffer_size"]:
                print("[Warning]. data remaining. queue size : ", queue.qsize())

            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                win_evaluation, score_evaluation = write_summary(writer, arg_dict, summary_queue, n_game, loss_lst,
                                                                 pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst,
                                                                 gail_pi_loss_lst, gail_exp_loss_lst, gail_pi_acc_list,
                                                                 gail_exp_acc_list, optimization_step,
                                                                 win_evaluation, score_evaluation)
                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
                n_game += arg_dict["summary_game_window"]

            _ = signal_queue.get()

        else:
            time.sleep(0.1)


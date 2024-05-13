import copy
import time
import random
import os
import importlib
from collections import deque
import numpy as np
import torch
from torch.distributions import Categorical

import gfootball.env as football_env


def get_action(a_prob, m_prob):
    a = Categorical(a_prob).sample().item()
    m, need_m = 0, 0
    prob_selected_a = a_prob[0][0][a].item()
    prob_selected_m = 0.5
    if a == 0:
        real_action = a
        prob = prob_selected_a
    elif a == 1:
        m = Categorical(m_prob).sample().item()
        need_m = 1
        real_action = m + 1
        prob_selected_m = m_prob[0][0][m].item()
        prob = prob_selected_a * prob_selected_m
    else:
        real_action = a + 7
        prob = prob_selected_a

    assert prob != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a, m, prob_selected_a, prob_selected_m)

    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m


def worker(worker_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print(f'[Worker {worker_num}] process started')

    imported_encoder = importlib.import_module("encoders." + arg_dict["encoder"])
    fe = imported_encoder.FeatureEncoder()
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])

    cpu_device = torch.device('cpu')
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())

    env = football_env.create_environment(env_name=arg_dict["env"], representation="raw", stacked=False,
                                          logdir='/tmp/football',
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)

    # Init
    n_epi = 0
    rollout = []
    pre_obs_deq = deque(maxlen=10)
    episode = deque(maxlen=3050)
    while True:  # episode loop
        env.reset()
        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float),
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))

        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        obs = env.observation()

        while not done:  # step loop
            init_t = time.time()

            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t

            h_in = h_out
            state_dict = fe.encode(obs[0])
            state_dict_tensor = fe.state_to_tensor(state_dict, h_in)

            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
            forward_t += time.time() - t1
            real_action, a, m, need_m, prob, prob_a, prob_m = get_action(a_prob, m_prob)

            pre_obs_deq.append(obs[0])
            obs, rew, done, info = env.step(real_action)
            fin_r = rewarder.calc_reward(rew, obs[0], pre_obs_deq)
            state_prime_dict = fe.encode(obs[0])
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            round_done = True if done or rew != 0. else False
            transition = (state_dict, real_action, a, m, fin_r, state_prime_dict, prob, prob_a, prob_m, round_done, need_m)
            episode.append(transition)

            steps += 1
            score += rew
            tot_reward += fin_r

            loop_t += time.time() - init_t

            if round_done:
                # calc return
                rewarder.calc_rtgs(episode, arg_dict["gamma"])  # add rtgs

                # send rollout
                for _ in range(len(episode)):
                    rollout.append(episode.popleft())
                    if len(rollout) == arg_dict["rollout_len"]:
                        data_queue.put(rollout)
                        rollout = []
                model.load_state_dict(center_model.state_dict())

            if done:
                if score > 0:
                    win = 1
                print(f"[TRAJ] score: {score}, total reward: {tot_reward:.2f}, loop time: {round(loop_t/steps, 4)}, wait time: {round(wait_t/steps, 4)}")
                summary_data = (win, score, tot_reward, steps, 0, loop_t / steps, forward_t / steps, wait_t / steps)
                summary_queue.put(summary_data)


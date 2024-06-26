import time
import importlib
import torch
from collections import deque
from torch.distributions import Categorical
import gfootball.env as football_env

from config import BASEDIR


def get_action(a_prob, m_prob):
    """
    :param a_prob:  all action probs
    :param m_prob:  all move probs
    :return:
    """
    a = Categorical(a_prob).sample().item()
    m, need_m = 0, 0
    prob_selected_a = a_prob[0][0][a].item()
    prob_selected_m = 0
    if a == 0:
        real_action = a
        prob_am = prob_selected_a
    elif a == 1:
        m = Categorical(m_prob).sample().item()
        need_m = 1
        real_action = m + 1
        prob_selected_m = m_prob[0][0][m].item()
        prob_am = prob_selected_a * prob_selected_m
    else:
        real_action = a + 7
        prob_am = prob_selected_a

    assert prob_am != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a, m, prob_selected_a, prob_selected_m)

    return real_action, a, m, need_m, prob_am, prob_selected_a, prob_selected_m


def evaluator(center_model, signal_queue, summary_queue, arg_dict):
    print("[Evaluator] process started")
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])

    fe = fe_module.FeatureEncoder()
    model = center_model

    env = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False,
                                          logdir='/tmp/football', write_goal_dumps=False,
                                          write_full_episode_dumps=False, render=False)
    n_epi = 0
    pre_obs_deq = deque(maxlen=30)
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
                # model.load_state_dict(center_model.state_dict())
                pass
            wait_t += time.time() - init_t

            h_in = h_out
            state_dict = fe.encode(obs[0])
            state_dict_tensor = fe.state_to_tensor(state_dict, h_in)

            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
            forward_t += time.time() - t1
            real_action, a, m, need_m, prob_am, prob_a, prob__m = get_action(a_prob, m_prob)

            pre_obs_deq.append(obs[0])
            prev_obs = obs
            obs, rew, done, info = env.step(real_action)
            fin_r = rewarder.calc_reward(rew, obs[0], pre_obs_deq)
            state_prime_dict = fe.encode(obs[0])

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

            steps += 1
            score += rew
            tot_reward += fin_r

            loop_t += time.time() - init_t

            if done:
                if score > 0:
                    win = 1
                print(f"[EVA] score: {score}, total reward: {tot_reward:.2f}")
                summary_data = (
                win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t / steps, forward_t / steps,
                wait_t / steps)
                summary_queue.put(summary_data)

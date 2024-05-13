import collections

import numpy as np


def calc_rtgs(episode: collections.deque, gamma: float) -> None:
    l = len(episode)
    discounted_r = 0.
    for _ in range(l):
        state_dict, real_action, a, m, fin_r, state_prime_dict, prob, prob_a, prob_m, done, need_m = episode.pop()
        discounted_r = fin_r + gamma * discounted_r
        # record.append(f"r: {fin_r}, rt: {discounted_r}")
        transition = (state_dict, real_action, a, m, fin_r, discounted_r, state_prime_dict, prob, prob_a, prob_m, done, need_m)
        episode.appendleft(transition)
    assert len(episode) == l, f"[Error] episode length: {len(episode)}"


def calc_reward(rew, obs, pre_obs_deq):
    ball_x, ball_y, ball_z = obs['ball']
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    obs_list = list(pre_obs_deq)
    prev_obs = obs_list[-1]

    ball_position_r = 0.0
    if (-END_X <= ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y < PENALTY_Y):  # 我方禁区
        ball_position_r = -2.0
    elif (-END_X <= ball_x < -MIDDLE_X) and (-END_Y < ball_y < END_Y):  # 左侧区域
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x <= MIDDLE_X) and (-END_Y < ball_y < END_Y):  # 中线区域
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x <= END_X) and (-PENALTY_Y < ball_y < PENALTY_Y):  # 敌方禁区
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x <= END_X) and (-END_Y < ball_y < END_Y):  # 右侧区域
        ball_position_r = 1.0
    else:  # 其他区域
        ball_position_r = 0.0

    # yellow_card
    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(prev_obs["left_team_yellow_card"])
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(prev_obs["right_team_yellow_card"])
    yellow_r = right_yellow - left_yellow

    # winner
    win_reward = 0.0
    if obs['steps_left'] == 0:
        [my_score, opponent_score] = obs['score']
        if my_score > opponent_score:
            win_reward = 1.0

    # total
    score_reward = rew  # + 0.1 * assist_goal
    reward = 5.0 * score_reward + 0.003 * ball_position_r  # + 0.03 * intercept


    # print(f'intercept: {intercept * 0.1},'
    #       f'ball_position_r: {ball_position_r * 3e-3},'
    #       f'yellow_r: {yellow_r},'
    #       f'rew: {rew * 5},'
    #       f'win_reward:{win_reward * 5}')

    return reward

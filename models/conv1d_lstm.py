import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Model(nn.Module):

    """
        player ->                      FC ->                       FC -> FC -> Policy(Action)
        ball ->                        FC ->
        left_team ->  FC -> conv1d ->  FC -> FC(Concat) -> LSTM -> FC -> FC -> Policy(Dir)
        right_team -> FC -> conv1d ->  FC ->
        left_closest ->                FC ->                       FC -> FC -> Value
        right_closest ->               FC ->
    """

    def __init__(self, arg_dict, device=None):
        super(Model, self).__init__()
        self.device = None
        if device:
            self.device = device

        self.arg_dict = arg_dict

        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player"], 64)
        self.fc_ball = nn.Linear(arg_dict["feature_dims"]["ball"], 64)
        self.fc_left = nn.Linear(arg_dict["feature_dims"]["left_team"], 48)
        self.fc_right = nn.Linear(arg_dict["feature_dims"]["right_team"], 48)
        self.fc_left_closest = nn.Linear(arg_dict["feature_dims"]["left_team_closest"], 48)
        self.fc_right_closest = nn.Linear(arg_dict["feature_dims"]["right_team_closest"], 48)

        self.conv1d_left = nn.Conv1d(48, 36, 1, stride=1)
        self.conv1d_right = nn.Conv1d(48, 36, 1, stride=1)
        self.fc_left2 = nn.Linear(36 * 10, 96)
        self.fc_right2 = nn.Linear(36 * 11, 96)
        self.fc_cat = nn.Linear(96 + 96 + 64 + 64 + 48 + 48, arg_dict["lstm_size"])

        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(48)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_left_closest = nn.LayerNorm(48)
        self.norm_right = nn.LayerNorm(48)
        self.norm_right2 = nn.LayerNorm(96)
        self.norm_right_closest = nn.LayerNorm(48)
        self.norm_cat = nn.LayerNorm(arg_dict["lstm_size"])

        self.lstm = nn.LSTM(arg_dict["lstm_size"], arg_dict["lstm_size"])

        self.fc_pi_a1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_a2 = nn.Linear(164, 12)
        self.norm_pi_a1 = nn.LayerNorm(164)

        self.fc_pi_m1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_m2 = nn.Linear(164, 8)
        self.norm_pi_m1 = nn.LayerNorm(164)

        self.fc_v1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.norm_v1 = nn.LayerNorm(164)
        self.fc_v2 = nn.Linear(164, 1, bias=False)
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])

    def forward(self, state_dict):
        player_state = state_dict["player"]
        ball_state = state_dict["ball"]
        left_team_state = state_dict["left_team"]
        left_closest_state = state_dict["left_closest"]
        right_team_state = state_dict["right_team"]
        right_closest_state = state_dict["right_closest"]
        avail = state_dict["avail"]

        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        left_team_embed = self.norm_left(self.fc_left(left_team_state))  # horizon, batch, n, dim
        left_closest_embed = self.norm_left_closest(self.fc_left_closest(left_closest_state))
        right_team_embed = self.norm_right(self.fc_right(right_team_state))
        right_closest_embed = self.norm_right_closest(self.fc_right_closest(right_closest_state))

        [horizon, batch_size, n_player, dim] = left_team_embed.size()
        left_team_embed = left_team_embed.view(horizon * batch_size, n_player, dim).permute(0, 2, 1)  # horizon * batch, dim1, n
        left_team_embed = F.relu(self.conv1d_left(left_team_embed)).permute(0, 2, 1)  # horizon * batch, n, dim2
        left_team_embed = left_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size,-1)  # horizon, batch, n * dim2
        left_team_embed = F.relu(self.norm_left2(self.fc_left2(left_team_embed)))

        right_team_embed = right_team_embed.view(horizon * batch_size, n_player + 1, dim).permute(0, 2, 1)  # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(0, 2, 1)  # horizon * batch, n * dim2
        right_team_embed = right_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size, -1)
        right_team_embed = F.relu(self.norm_right2(self.fc_right2(right_team_embed)))

        cat = torch.cat(
            [player_embed, ball_embed, left_team_embed, right_team_embed, left_closest_embed, right_closest_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))  # horizon, batch, 256
        h_in = state_dict["hidden"]
        out, h_out = self.lstm(cat, h_in)

        a_out = F.relu(self.norm_pi_a1(self.fc_pi_a1(out)))
        a_out = self.fc_pi_a2(a_out)
        logit = a_out + (avail - 1) * 1e7
        prob = F.softmax(logit, dim=2)

        prob_m = F.relu(self.norm_pi_m1(self.fc_pi_m1(out)))
        prob_m = self.fc_pi_m2(prob_m)
        prob_m = F.softmax(prob_m, dim=2)

        v = F.relu(self.norm_v1(self.fc_v1(out)))
        v = self.fc_v2(v)

        return prob, prob_m, v, h_out

    def make_batch(self, data) -> list:  # todo list->numpy->tensor
        # data shape: (batchsize, horizon, datalist)
        s_player_batch, s_ball_batch, s_left_batch, s_left_closest_batch, s_right_batch, s_right_closest_batch, avail_batch = [], [], [], [], [], [], []
        s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch, s_left_closest_prime_batch, \
        s_right_prime_batch, s_right_closest_prime_batch, avail_prime_batch = [], [], [], [], [], [], []
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        real_a_batch, a_batch, m_batch, r_batch, rt_batch, prob_batch, prob_a_batch, prob_m_batch, done_batch, need_move_batch = [], [], [], [], [], [], [], [], [], []

        for rollout in data:
            s_player_lst, s_ball_lst, s_left_lst, s_left_closest_lst, s_right_lst, s_right_closest_lst, avail_lst = [], [], [], [], [], [], []
            s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, s_left_closest_prime_lst, \
            s_right_prime_lst, s_right_closest_prime_lst, avail_prime_lst = [], [], [], [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            real_a_list, a_lst, m_lst, r_lst, rt_lst, prob_lst, prob_a_lst, prob_m_lst, done_lst, need_move_lst = [], [], [], [], [], [], [], [], [], []

            for transition in rollout:
                s, real_a, a, m, r, rt, s_prime, prob, prob_a, prob_m, done, need_move = transition

                s_player_lst.append(s["player"])
                s_ball_lst.append(s["ball"])
                s_left_lst.append(s["left_team"])
                s_left_closest_lst.append(s["left_closest"])
                s_right_lst.append(s["right_team"])
                s_right_closest_lst.append(s["right_closest"])
                avail_lst.append(s["avail"])
                h1_in, h2_in = s["hidden"]
                h1_in_lst.append(h1_in)
                h2_in_lst.append(h2_in)

                s_player_prime_lst.append(s_prime["player"])
                s_ball_prime_lst.append(s_prime["ball"])
                s_left_prime_lst.append(s_prime["left_team"])
                s_left_closest_prime_lst.append(s_prime["left_closest"])
                s_right_prime_lst.append(s_prime["right_team"])
                s_right_closest_prime_lst.append(s_prime["right_closest"])
                avail_prime_lst.append(s_prime["avail"])
                h1_out, h2_out = s_prime["hidden"]
                h1_out_lst.append(h1_out)
                h2_out_lst.append(h2_out)

                real_a_list.append([real_a])
                a_lst.append([a])
                m_lst.append([m])
                r_lst.append([r])
                rt_lst.append([rt])
                prob_lst.append([prob])
                prob_a_lst.append([prob_a])
                prob_m_lst.append([prob_m])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                need_move_lst.append([need_move]),

            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_left_batch.append(s_left_lst)
            s_left_closest_batch.append(s_left_closest_lst)
            s_right_batch.append(s_right_lst)
            s_right_closest_batch.append(s_right_closest_lst)
            avail_batch.append(avail_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])

            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_left_closest_prime_batch.append(s_left_closest_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)
            s_right_closest_prime_batch.append(s_right_closest_prime_lst)
            avail_prime_batch.append(avail_prime_lst)
            h1_out_batch.append(h1_out_lst[0])
            h2_out_batch.append(h2_out_lst[0])

            real_a_batch.append(real_a_list)
            a_batch.append(a_lst)
            m_batch.append(m_lst)
            r_batch.append(r_lst)
            rt_batch.append(rt_lst)
            prob_batch.append(prob_lst)
            prob_a_batch.append(prob_a_lst)
            prob_m_batch.append(prob_m_lst)
            done_batch.append(done_lst)
            need_move_batch.append(need_move_lst)

        s = {
            "player": torch.tensor(s_player_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "right_team": torch.tensor(s_right_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "avail": torch.tensor(avail_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2))
        }

        s_prime = {
            "player": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "right_team": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "avail": torch.tensor(avail_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2))
        }

        real_a = torch.tensor(real_a_batch, device=self.device).permute(1, 0, 2)
        a = torch.tensor(a_batch, device=self.device).permute(1, 0, 2)
        m = torch.tensor(m_batch, device=self.device).permute(1, 0, 2)
        r = torch.tensor(r_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)
        rt = torch.tensor(rt_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)
        done_mask = torch.tensor(done_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)
        prob = torch.tensor(prob_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)
        prob_a = torch.tensor(prob_a_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)
        prob_m = torch.tensor(prob_m_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)
        need_move = torch.tensor(need_move_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)

        return [s, real_a, a, m, r, rt, s_prime, done_mask, prob, prob_a, prob_m, need_move]

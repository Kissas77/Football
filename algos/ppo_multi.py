"""
    optim
    1 MC return -> GAE
    2 add entropy
    3 add clip_grad_norm
    4 add advantage normalization
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Algo():
    def __init__(self, arg_dict, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = arg_dict["eps_clip"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]

    def train(self, model, data):
        tot_loss_lst = []
        pi_loss_lst = []
        entropy_lst = []
        move_entropy_lst = []
        v_loss_lst = []

        # to calculate fixed advantages before update
        data_with_adv = []
        # data shape: (buffer_size, trans_len, rollout_len, batch_size, *)
        for mini_batch in data:  # mini_batch: [s, real_a, a, m, r, s_prime, done_mask, prob, need_move]
            s, real_a, a, m, r, rt, s_prime, done_mask, prob, prob_a, prob_m, need_move = mini_batch
            with torch.no_grad():  # input shape:(rollout_len, batch_size, *)
                pi, pi_move, v, _ = model(s)  # output shape:(rollout_len, batch_size, 1)
                pi_prime, pi_m_prime, v_prime, _ = model(s_prime)

            td_target = r + self.gamma * v_prime * done_mask  # done_mask==0
            delta = td_target - v
            delta = delta.detach().cpu().numpy()  # shape: (rollout_len, batch_size, 1)
            done_mask_ = done_mask.cpu().numpy()
            adv_lst = []
            adv_tmp = np.array([0])
            for i in range(len(delta) - 1, -1, -1):
                adv_tmp = delta[i] + self.gamma * self.lmbda * adv_tmp * done_mask_[i]
                adv_lst.append(adv_tmp)
            adv_lst.reverse()  # shape: (rollout_len, batch_size, 1)
            advantage = torch.tensor(adv_lst, dtype=torch.float, device=model.device)
            # rtgs
            rtgs = advantage + v.detach()
            # Trick: normalize advantage
            adv = (advantage - advantage.mean(dim=(0, 1), keepdim=True)) / (advantage.std(dim=(0, 1), keepdim=True) + 1e-10)

            # data_with_adv shape: (buffer_size, trans_len, rollout_len, batch_size, *)
            data_with_adv.append((s, a, m, r, s_prime, done_mask, prob, prob_a, prob_m, need_move, rtgs, adv))

        for i in range(self.K_epoch):
            for mini_batch in data_with_adv:  # mini_batch shape: (trans_len, rollout_len, batch_size, *)
                # samples nums: rollout_len * batch_size
                s, a, m, r, s_prime, done_mask, prob, prob_a, prob_m, need_move, rtgs, adv = mini_batch
                pi, pi_move, v, _ = model(s)  # pi shape: (horizon, batch_size, action_len)

                # policy loss
                pi_a = pi.gather(2, a)  # pi_a shape: (horizon, batch_size, 1)
                pi_m = pi_move.gather(2, m)  # pi_m shape: (horizon, batch_size, 1)
                ratio_a = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
                ratio_m = need_move * torch.exp(torch.log(pi_m) - torch.log(prob_m))
                ratio = torch.cat((ratio_a, ratio_m), dim=2)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv
                surr_loss = -torch.min(surr1, surr2)

                # entropy loss
                # entropy_a = -torch.sum(pi*torch.log(pi), dim=2, keepdim=True)
                entropy_a = -torch.log(pi_a)
                # entropy_m = -need_move * torch.sum(pi_move*torch.log(pi_move), dim=2, keepdim=True)
                entropy_m = -need_move * torch.log(pi_m)
                entropy = torch.cat((entropy_a, entropy_m), dim=2)
                entropy_loss = -1 * self.entropy_coef * entropy

                # value loss
                v_loss = F.smooth_l1_loss(v, rtgs)

                # total loss
                loss = surr_loss.mean() + v_loss + entropy_loss.mean()

                # optimize model
                model.optimizer.zero_grad()
                loss.backward()
                # Trick: clip grad
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                model.optimizer.step()

                # tensorboard
                tot_loss_lst.append(loss.item())
                pi_loss_lst.append(surr_loss.mean().item())
                v_loss_lst.append(v_loss.item())
                entropy_lst.append(entropy.mean().item())
                n_need_move = torch.sum(need_move).item()
                if n_need_move == 0:
                    move_entropy_lst.append(0)
                else:
                    move_entropy_lst.append((torch.sum(entropy_m)/n_need_move).item())
        return np.mean(tot_loss_lst), np.mean(pi_loss_lst), np.mean(v_loss_lst), np.mean(entropy_lst), np.mean(move_entropy_lst)

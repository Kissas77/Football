import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Algo():
    def __init__(self, arg_dict, device=None):
        self.gail_epoch = arg_dict["gail_epoch"]
        # self.reg_coef = arg_dict["gali_reg_coef"]

    def train(self, discriminator, data_pi, data_exp) -> tuple:
        exp_loss_list = []
        pi_loss_list = []
        pi_acc_list = []
        exp_acc_list = []
        reg_lst = []

        for i in range(self.gail_epoch):
            for mini_batch_pi, mini_batch_exp in zip(data_pi, data_exp):
                # mini_batch_pi, mini_batch_exp = data_pi, data_exp
                s, real_a, a, m, r, s_prime, done_mask, prob, need_move = mini_batch_pi
                s_exp, a_exp, done_exp = mini_batch_exp  # (batch(rollout(transitaion)))
                logits_pi = discriminator(s, real_a)
                logits_exp = discriminator(s_exp, a_exp)

                loss_pi = F.binary_cross_entropy(logits_pi, torch.zeros_like(logits_pi))
                loss_exp = F.binary_cross_entropy(logits_exp, torch.ones_like(logits_exp))
                # reg
                # r1_reg = 0.
                # for param in discriminator.parameters():
                #     grad = param.grad
                #     r1_reg += param.grad.norm()
                loss_disc = loss_pi + loss_exp  # todo decay loss_pi

                discriminator.optimizer.zero_grad()
                # (loss_disc + self.reg_coef * r1_reg).backward()
                loss_disc.backward()
                discriminator.optimizer.step()

                pi_loss_list.append(loss_pi.item())
                exp_loss_list.append(loss_exp.item())
                # reg_lst.append(r1_reg.item()

                pi_acc_list.append((logits_pi > 0.5).float().mean().item())
                exp_acc_list.append((logits_exp > 0.5).float().mean().item())

        return np.mean(pi_loss_list), np.mean(exp_loss_list), np.mean(pi_acc_list), np.mean(exp_acc_list)

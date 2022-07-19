import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 
import numpy as np
import sys 
from collections import deque

class Sar():
    """
    Style-Agnostic Reinforcement Learning (SAR)
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr,
                 eps,
                 max_grad_norm,
                 aug_func=None,
                 adv_coef=0.1,
                 val_sim_coef=0.1):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.step = 0
        self.aug_func = aug_func
        self.adv_coef = adv_coef
        self.val_sim_coef = val_sim_coef
        
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.optimizer_act = optim.Adam(actor_critic.act_parameters(), lr=lr, eps=eps)
        self.optimizer_adv = optim.Adam(actor_critic.adv_parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        self.step += 1

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        kl_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                
                obs_batch = self.aug_func.do_augmentation(obs_batch)

                values, action_log_probs, dist_entropy, logits, rnn_hxs, \
                    value_adv, logits_adv = self.actor_critic.evaluate_actions(
                                                obs_batch, recurrent_hidden_states_batch, \
                                                masks_batch, actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                value_loss_aug = 0.5 * (torch.detach(values) - value_adv).pow(2).mean()

                kl_loss = self.kl_loss(logits.log(), logits_adv)

                # Update actor-critic using both PPO Loss
                self.optimizer_act.zero_grad()
                loss = action_loss + \
                        self.value_loss_coef * value_loss \
                        + self.val_sim_coef * value_loss_aug\
                        - self.entropy_coef * dist_entropy \
                        + self.adv_coef * kl_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer_act.step()  


                values, action_log_probs, dist_entropy, logits, rnn_hxs, \
                    value_adv, logits_adv = self.actor_critic.evaluate_actions(
                                                obs_batch, recurrent_hidden_states_batch, \
                                                masks_batch, actions_batch)
                kl_loss = self.kl_loss(logits.log(), logits_adv)

                # update adv attacker
                self.optimizer_adv.zero_grad()
                loss = - kl_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer_adv.step()  

                 
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                kl_loss_epoch += kl_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kl_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kl_loss_epoch
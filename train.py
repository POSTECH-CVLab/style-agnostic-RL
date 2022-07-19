import copy
import os
import time
from collections import deque

import numpy as np
import torch

import common.utils as utils
import common.data_augs as data_augs
from common.storage import RolloutStorage
from baselines import logger

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from common.envs import VecPyTorchProcgen, TransposeImageProcgen

from common.arguments import parser
from test import evaluate

aug_to_func = {    
        'crop': data_augs.Crop,
        'cutoutcolor': data_augs.CutoutColor,

        'flip': data_augs.Flip,
        'rotate': data_augs.Rotate,
        'cutout': data_augs.Cutout,
        'grayscale': data_augs.Grayscale,
        'colorjitter': data_augs.ColorJitter,
        'random-conv': data_augs.RandomConv,
}

def train(args):
    # hw settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.gpu_device}" if args.cuda else "cpu")

    # environment
    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name, \
        num_levels=args.num_levels, start_level=args.start_level, \
        distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)
    
    # agent
    obs_shape = envs.observation_space.shape
    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    if args.algo == "ppo":
        from common.policy import PpoPolicy as Policy
        from agents.ppo import Ppo as Agent
        policy_kwargs={
            'recurrent': False, 'hidden_size': args.hidden_size, "encoding": False}
        agent_kwargs = {}

    elif args.algo == "sar":
        from common.policy import SarPolicy as Policy
        from agents.sar import Sar as Agent
        policy_kwargs={
            'recurrent': False, 'hidden_size': args.hidden_size, \
                "n_envs": args.num_processes, "encoding": False}
        agent_kwargs = {
            "aug_func": aug_to_func[args.aug_type](batch_size=batch_size, \
                            gpu_device=device) if args.aug_type != "identity" else data_augs.Identity(),
            "adv_coef": args.adv_coef, "val_sim_coef": args.val_sim_coef}

    actor_critic = Policy(
        obs_shape,
        envs.action_space.n,
        policy_kwargs).to(device)  

    agent = Agent(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.lr,
        args.eps,
        args.max_grad_norm,
        **agent_kwargs)

    # Settings
    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)
    if "aug_func" in agent_kwargs.keys():
        if args.algo == "sar":
            log_file = f'-{args.env_name}-{args.algo}({args.aug_type})(adv{args.adv_coef})(val{args.val_sim_coef})-seed{args.seed}'
        else:
            log_file = f'-{args.env_name}-{args.algo}({args.aug_type})-seed{args.seed}'
    else:
        log_file = f'-{args.env_name}-{args.algo}-seed{args.seed}'
    checkpoint_path = os.path.join(args.save_dir, "agent" + log_file + ".pt")
    if os.path.exists(checkpoint_path) and args.preempt:
        checkpoint = torch.load(checkpoint_path)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        if args.algo == "sar":
            agent.optimizer_act.load_state_dict(checkpoint['optimizer_act_state_dict'])
            agent.optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
        else:
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch'] + 1
        logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file + "-e%s" % init_epoch)
    else:
        init_epoch = 0
        logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file)


    # storage
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size,
                                aug_type=args.aug_type,
                                algo=args.algo, num_level=args.num_levels)


    ## Main training loop ##
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(init_epoch, num_updates):
        actor_critic.train()
        for step in range(args.num_steps):
            # take actions
            with torch.no_grad():
                if args.algo == "sar":
                    value, action, action_log_prob, logits, recurrent_hidden_states, \
                        _, _   = actor_critic.act(
                                                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                                                rollouts.masks[step])
                else:
                    value, action, action_log_prob, logits, \
                        recurrent_hidden_states = actor_critic.act(
                                                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                                                rollouts.masks[step])
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # store to rollouts
            # if done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action, \
                                action_log_prob, value, reward, masks, bad_masks)
        with torch.no_grad():
            obs_id = rollouts.obs[-1]
            next_value = actor_critic.get_value(
                obs_id, rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # optimize the policy
        if args.algo == "sar":
            value_loss, action_loss, dist_entropy, kl_loss = agent.update(rollouts)    
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)    
        rollouts.after_update()

        # log the outputs
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("\nUpdate {}, step {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}"
                .format(j, total_num_steps,
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            
            logger.logkv("train/nupdates", j)
            logger.logkv("train/total_num_steps", total_num_steps)            

            logger.logkv("losses/dist_entropy", dist_entropy)
            logger.logkv("losses/value_loss", value_loss)
            logger.logkv("losses/action_loss", action_loss)
            if args.algo == "sar":
                logger.logkv("losses/kl_loss", kl_loss)
            
            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))

            ### Eval on the Full Distribution of Levels ###
            eval_episode_rewards = evaluate(args, actor_critic, device)

            logger.logkv("test/mean_episode_reward", np.mean(eval_episode_rewards))
            logger.logkv("test/median_episode_reward", np.median(eval_episode_rewards))

            logger.dumpkvs()

        # save model
        if (j > 0 and j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass
            
            if args.algo == "sar":
                torch.save({
                        'epoch': j,
                        'model_state_dict': agent.actor_critic.state_dict(),
                        'optimizer_act_state_dict': agent.optimizer_act.state_dict(),
                        'optimizer_adv_state_dict': agent.optimizer_adv.state_dict(),
                }, os.path.join(args.save_dir, "agent" + log_file + ".pt")) 
            else:
                torch.save({
                        'epoch': j,
                        'model_state_dict': agent.actor_critic.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                }, os.path.join(args.save_dir, "agent" + log_file + ".pt")) 

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)

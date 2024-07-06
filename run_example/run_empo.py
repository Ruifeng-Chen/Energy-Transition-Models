import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, MLPETM
from offlinerlkit.dynamics import EnergyDynamics, EnsembleEnergyDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import EMPOPolicy




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="empo")
    parser.add_argument("--task", type=str, default="walker2d-medium-v2")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--penalty-coef", type=float, default=2.0)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)   
    parser.add_argument("--deterministic-backup", type=bool, default=False)   
    parser.add_argument("--target_zero_clip", type=bool, default=False)  
    parser.add_argument("--penalty-decay", type=bool, default=True) 

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--penalty-type", type=str, default="ensemble_std", choices=["ensemble_std"])

    parser.add_argument("--etm_lr", type=float, default=1e-3)
    parser.add_argument("--etm_hidden_dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--etm_activation", type=str, default="relu")
    parser.add_argument("--etm_with_reward", type=bool, default=True)
    parser.add_argument("--etm_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--etm_num_negative_samples", type=int, default=16)
    parser.add_argument("--etm_loss_type", type=str, default="info_nce")
    parser.add_argument("--etm_add_grad_penalty", type=bool, default=True)
    parser.add_argument("--etm_grad_penalty_margin", type=float, default=5.0)
    parser.add_argument("--etm_max_epochs", type=int, default=200)
    parser.add_argument("--etm_batch_size", type=int, default=1024)
    parser.add_argument("--etm_step_noise", type=float, default=0.5)  
    parser.add_argument("--etm_step_iter", type=int, default=30)    

    parser.add_argument("--load_etm_path", type=str, nargs='*', default=[
        "log/walker2d-medium-v2/etm/seed_1&timestamp_23-1116-223315/model",
        "log/walker2d-medium-v2/etm/seed_2&timestamp_23-1117-150227/model",
        "log/walker2d-medium-v2/etm/seed_3&timestamp_23-1121-144644/model",
        "log/walker2d-medium-v2/etm/seed_4&timestamp_23-1210-001908/model",
        "log/walker2d-medium-v2/etm/seed_5&timestamp_23-1214-131232/model",
    ])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    print(args.task)
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    termination_fn = get_termination_fn(task=args.task)


    if args.load_etm_path:
        etm_list = []
        for path in args.load_etm_path:
            energy_model = MLPETM(
                obs_dim=np.prod(args.obs_shape),
                action_dim=args.action_dim,
                hidden_dims=args.etm_hidden_dims,
                activation=args.etm_activation,
                with_reward=args.etm_with_reward,
                device=args.device
            )
            etm_optim = torch.optim.Adam(
                energy_model.parameters(),
                lr=args.etm_lr,
                weight_decay=1e-4,
            )
            scaler = StandardScaler()
            delta_scaler = StandardScaler(box_normalization=True)

            etm_dynamics = EnergyDynamics(
                energy_model,
                etm_optim,
                scaler,
                delta_scaler,
                termination_fn,

                softmax_temperature=args.etm_softmax_temperature,
                num_negative_samples=args.etm_num_negative_samples,

                etm_loss_type=args.etm_loss_type,
                add_grad_penalty=args.etm_add_grad_penalty,
                grad_penalty_margin=args.etm_grad_penalty_margin,

                step_noise=args.etm_step_noise,
                step_iter=args.etm_step_iter,
            )
            etm_dynamics.load(path)


            etm_list.append(etm_dynamics)

        ensemble_energy_dynamics = EnsembleEnergyDynamics(etm_list, args.penalty_coef, args.penalty_type)


    # create policy
    policy = EMPOPolicy(
        ensemble_energy_dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_type=args.penalty_type,
        penalty_coef=args.penalty_coef,
        penalty_decay=args.penalty_decay,
        deterministic_backup=args.deterministic_backup,
        target_zero_clip=args.target_zero_clip,
    )

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device,
        save_penalty=True if args.penalty_type == "ensemble_std" else False,
    )

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["penalty_coef", "rollout_length"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )
    
    policy_trainer.train()


if __name__ == "__main__":
    train()
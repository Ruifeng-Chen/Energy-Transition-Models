import os
import gym
import d4rl
from gym import spaces
from gym.core import Env
import random
import pickle
import argparse
import numpy as np
import tensorflow as tf
import tree
import torch

import sys
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.modules import EnsembleDynamicsModel
from offlinerlkit.modules.etm_module import MLPETM
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.dynamics import EnergyDynamics, EnsembleDynamics
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.logger import Logger, make_log_dirs, load_args



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="walker2d-medium-v2")

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--load_dynamics_path", type=str, default=None)

    parser.add_argument("--load_etm_path", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

# class AntWrapper(gym.ObservationWrapper):

#     def __init__(self, env: Env):
#         super().__init__(env)
#         self._observation_space = spaces.Box(-float("inf"), float("inf"), (27,),  dtype=np.float32)

#     def observation(self, observation):
#         obs = observation[:27]
#         return obs

class D4RLPolicy:
    """D4RL policy."""

    # def __init__(self, policy_file, device="cpu"):
    def __init__(self, task: str, index: int, device="cpu"):
        self.device = torch.device(device)
        dir_path = "ope_policy"
        # with tf.io.gfile.GFile(os.path.join('gs://gresearch/deep-ope/d4rl', policy_file), 'rb') as f:
        if task in ["hopper", "halfcheetah", "walker", "ant"]:
            policy_file = "{0}/deep-ope_d4rl_{0}_{0}_online_{1}.pkl".format(task, index)
        elif task in ["door", "pen", "hammer", "relocate"]:
            policy_file = "{0}/deep-ope_d4rl_{0}_{0}_dapg_{1}.pkl".format(task, index)

        with tf.io.gfile.GFile(os.path.join(dir_path, policy_file), 'rb') as f:
            weights = pickle.load(f)
        self.fc0_w = torch.from_numpy(weights['fc0/weight']).to(self.device)
        self.fc0_b = torch.from_numpy(weights['fc0/bias']).to(self.device)
        self.fc1_w = torch.from_numpy(weights['fc1/weight']).to(self.device)
        self.fc1_b = torch.from_numpy(weights['fc1/bias']).to(self.device)
        self.fclast_w = torch.from_numpy(weights['last_fc/weight']).to(self.device)
        self.fclast_b = torch.from_numpy(weights['last_fc/bias']).to(self.device)
        self.fclast_w_logstd = torch.from_numpy(weights['last_fc_log_std/weight']).to(self.device)
        self.fclast_b_logstd = torch.from_numpy(weights['last_fc_log_std/bias']).to(self.device)
        # relu = lambda x: torch.maximum(x, 0)
        self.nonlinearity = torch.tanh if weights['nonlinearity'] == 'tanh' else torch.relu

        identity = lambda x: x
        self.output_transformation = torch.tanh if weights[
            'output_distribution'] == 'tanh_gaussian' else identity

    def select_action(self, state, deterministic=False):
        # if torch.is_tensor(state): state = state.cpu().numpy()
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        x = torch.mm(state, self.fc0_w.T) + self.fc0_b
        x = self.nonlinearity(x)
        x = torch.mm(x, self.fc1_w.T) + self.fc1_b
        x = self.nonlinearity(x)
        mean = torch.mm(x, self.fclast_w.T) + self.fclast_b
        logstd = torch.mm(x, self.fclast_w_logstd.T) + self.fclast_b_logstd
        if deterministic:
            action = self.output_transformation(mean)
        else:
            noise = torch.ones_like(logstd)
            action = self.output_transformation(mean + torch.exp(logstd) * noise)
        return action.cpu().numpy()

def mbope(dynamics, eval_policy_set, env, training_dataset=None, gamma=0.995, num_eval_episodes=10):
        print("============================================================")
        if isinstance(dynamics, EnergyDynamics):
            print("Evaluating policies in Energy-based Model")

        
        dynamics.set_for_eval_value_gap(env, eval_policy_set, training_dataset)
        real_values = dynamics.eval_real_values(gamma=gamma, num_eval_episodes=num_eval_episodes)
        fake_values = dynamics.eval_fake_values(gamma=gamma, num_eval_episodes=num_eval_episodes)

        real_values, fake_values = np.array(real_values), np.array(fake_values)
        value_min, value_max = real_values.min(), real_values.max()
        norm_real_values = (real_values - value_min) / (value_max - value_min)
        norm_fake_values = (fake_values - value_min) / (value_max - value_min)
        print(norm_real_values)
        print(norm_fake_values)
        raw_absolute_error = (np.abs(real_values - fake_values)).mean()
        absolute_error = (np.abs(norm_real_values - norm_fake_values)).mean()
        rank_correlation = np.corrcoef(norm_real_values, norm_fake_values)[0, 1]

        top_idxs = np.argsort(norm_fake_values)[-1:]
        top5_idxs = np.argsort(norm_fake_values)[-5:]
        regret = norm_real_values.max() - norm_real_values[top_idxs].max()
        regret5 = norm_real_values.max() - norm_real_values[top5_idxs].max()

        print("----------------------------------------------------------")
        print(f"raw absolute error: {raw_absolute_error}")
        print(f"absolute error: {absolute_error}")
        print(f"rank correlation: {rank_correlation}")
        print(f"regret: {regret}")
        print(f"regret@5: {regret5}")
        return {
            "raw absolute error": raw_absolute_error,
            "absolute_error": absolute_error,
            "rank correlation": rank_correlation,
            "regret": regret,
            "regret@5": regret5,
        }
       
def mbope_ensemble(dynamics: EnsembleDynamics, eval_policy_set, env, gamma=0.995, num_eval_episodes=10):
    print("============================================================")
    
    dynamics.set_for_eval_value_gap(env, eval_policy_set)
    real_values = dynamics.eval_real_values(gamma=gamma, num_eval_episodes=num_eval_episodes)

    raw_absolute_errors, absolute_errors, rank_correlations, regrets, regret5s = [], [], [], [], []

    # for idx in range(dynamics.model.num_ensemble):
    for idx in np.random.choice(dynamics.model.num_ensemble, 5, replace=False):

        fake_values = dynamics.eval_fake_values(gamma=gamma,num_eval_episodes=num_eval_episodes, model_idx=int(idx))
        real_values, fake_values = np.array(real_values), np.array(fake_values)
        value_min, value_max = real_values.min(), real_values.max()
        norm_real_values = (real_values - value_min) / (value_max - value_min)
        norm_fake_values = (fake_values - value_min) / (value_max - value_min)

        raw_absolute_error = (np.abs(real_values - fake_values)).mean()
        absolute_error = (np.abs(norm_real_values - norm_fake_values)).mean()
        rank_correlation = np.corrcoef(norm_real_values, norm_fake_values)[0, 1]

        top_idxs = np.argsort(norm_fake_values)[-1:]
        top5_idxs = np.argsort(norm_fake_values)[-5:]
        regret = norm_real_values.max() - norm_real_values[top_idxs].max()
        regret5 = norm_real_values.max() - norm_real_values[top5_idxs].max()

        print("----------------------------------------------------------")
        print(f"raw absolute error: {raw_absolute_error}")
        print(f"absolute error: {absolute_error}")
        print(f"rank correlation: {rank_correlation}")
        print(f"regret: {regret}")
        print(f"regret@5: {regret5}")
        print("----------------------------------------------------------")

        raw_absolute_errors.append(raw_absolute_error)
        absolute_errors.append(absolute_error)
        rank_correlations.append(rank_correlation)
        regrets.append(regret)
        regret5s.append(regret5)

    mean_raw_absolute_error = np.array(raw_absolute_errors).mean()
    std_raw_absolute_error = np.array(raw_absolute_errors).std()
    mean_absolute_error = np.array(absolute_errors).mean()
    std_absolute_error = np.array(absolute_errors).std()
    mean_rank_correlation = np.array(rank_correlations).mean()
    std_rank_correlation = np.array(rank_correlations).std()
    mean_regret = np.array(regrets).mean()
    std_regret = np.array(regrets).std()
    mean_regret5 = np.array(regret5s).mean()
    std_regret5 = np.array(regret5s).std()

    print("******************************************************************")
    print(f"raw absolute error: {mean_raw_absolute_error} +- {std_raw_absolute_error}")
    print(f"absolute error: {mean_absolute_error} +- {std_absolute_error}")
    print(f"rank correlation: {mean_rank_correlation} +- {std_rank_correlation}")
    print(f"regret: {mean_regret} +- {std_regret}")
    print(f"regret@5: {mean_regret5} +- {std_regret5}")


def main(args=get_args()):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    print(args.task)
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)

    # if args.task.split("-")[0] == "ant":
    #     # env = AntWrapper(env)
    #     dataset["observations"] = dataset["observations"][:, :27]
    #     dataset["next_observations"] = dataset["next_observations"][:, :27]

    print(dataset["observations"].shape, dataset["actions"].shape)

    task = args.task.split("-")[0]
    if task == "walker2d":
        task = "walker"
    eval_policy_set = [D4RLPolicy(task=task, index=idx, device=args.device) for idx in range(10)]

    env.seed(args.seed)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]


    if args.load_dynamics_path is not None:
        if os.path.basename(args.load_dynamics_path) == "model":
            fw_arg_path = os.path.join(os.path.dirname(args.load_dynamics_path), "record/hyper_param.json")
        fw_args = load_args(fw_arg_path)
        dynamics_model = EnsembleDynamicsModel(
            obs_dim=np.prod(args.obs_shape),
            action_dim=args.action_dim,
            hidden_dims=fw_args.dynamics_hidden_dims,
            num_ensemble=fw_args.n_ensemble,
            num_elites=fw_args.n_elites,
            weight_decays=fw_args.dynamics_weight_decay,
            device=args.device,
        )
        dynamics_optim = torch.optim.Adam(
            dynamics_model.parameters(),
            lr=fw_args.dynamics_lr,
        )
        scaler = StandardScaler()
        termination_fn = get_termination_fn(task=args.task)
        dynamics = EnsembleDynamics(
            dynamics_model,
            dynamics_optim,
            scaler,
            termination_fn,
        )

        dynamics.load(args.load_dynamics_path)

        ope_info = mbope_ensemble(dynamics, eval_policy_set, env)

    if args.load_etm_path is not None:
        if os.path.basename(args.load_etm_path) == "model":
            etm_arg_path = os.path.join(os.path.dirname(args.load_etm_path), "record/hyper_param.json")
        etm_args = load_args(etm_arg_path)

        energy_model = MLPETM(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=etm_args.etm_hidden_dims,
        activation=etm_args.etm_activation,
        with_reward=True,
        device=args.device
    )
        etm_optim = torch.optim.Adam(
            energy_model.parameters(),
            lr=etm_args.etm_lr ,
        )
        scaler = StandardScaler()
        delta_scaler = StandardScaler()
        termination_fn = get_termination_fn(task=args.task)
        etm_dynamics = EnergyDynamics(
            energy_model,
            etm_optim,
            scaler,
            delta_scaler,
            terminal_fn=termination_fn,
        )

        etm_dynamics.load(args.load_etm_path)

        # ope_etm_info = mbope(etm_dynamics, eval_policy_set, env, dataset)
        ope_etm_info = mbope(etm_dynamics, eval_policy_set, env)
    
if __name__ == "__main__":
    main()
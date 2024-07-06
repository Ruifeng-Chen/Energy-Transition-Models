import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from typing import Callable, List, Tuple, Dict, Optional, Union
from torch.utils.data.dataloader import DataLoader
from offlinerlkit.dynamics import BaseDynamics, EnsembleDynamics
from offlinerlkit.modules import MLPETM
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules.etm_module import MLPETM, langevin_mcmc_sa_s, grad_wrt_next_s


class EnergyDynamics():

    def __init__(
            self,
            energy_model: MLPETM,
            optim: torch.optim.Optimizer,
            scaler: StandardScaler,
            delta_scaler: StandardScaler,
            terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
            
            softmax_temperature: float = 1.0,
            num_negative_samples: int = 20,
            etm_loss_type: str = 'info_nce',
            add_grad_penalty: bool = True,
            grad_penalty_margin: float = 1.0,

            langevin_iterations: int = 50,
            langevin_noise_scale: float = 0.5,
            langevin_grad_clip: float = 10.,
            langevin_delta_clip: float = 0.5,
            langevin_stepsize_init: float = 0.1,
            langevin_stepsize_final: float = 1e-3,

            step_noise: float = 0.0,
            step_iter: int = 50,
        ) -> None:
        
        self.energy_model = energy_model
        self.device = self.energy_model.device
        self.saved_energy_model = []
        self.optim = optim
        self.scaler = scaler
        self.delta_scaler = delta_scaler
        self.terminal_fn = terminal_fn

        self.softmax_temperature = softmax_temperature
        self.num_negative_samples = num_negative_samples
        self.num_langevin_samples = self.num_negative_samples


        self.etm_loss_type = etm_loss_type
        self.add_grad_penalty = add_grad_penalty
        self.grad_penalty_margin = grad_penalty_margin

        self.langevin_iterations = langevin_iterations
        self.langevin_noise_scale = langevin_noise_scale
        self.langevin_grad_clip = langevin_grad_clip
        self.langevin_delta_clip = langevin_delta_clip
        self.langevin_stepsize_init = langevin_stepsize_init
        self.langevin_stepsize_final = langevin_stepsize_final

        self.step_noise = step_noise
        self.step_iter = step_iter

        self.next_obss_pred_lower, self.next_obss_pred_upper = None, None
        self.reward_pred_lower, self.reward_pred_upper = None, None


    def step(self, obs, act):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        if len(act.shape) == 1:
            act = act.reshape(1, -1)
        
        obs_act = np.concatenate([obs, act], axis=-1)
        obs_act = self.scaler.transform(obs_act)

        target_shape = list(obs.shape)
        if self.energy_model.with_reward:
            target_shape[1] += 1
        init_target = torch.zeros(target_shape, device=self.device)
        target, info = self.predict(obs_act, init_target, langevin_steps=self.step_iter, langevin_noise=self.step_noise)
        if self.energy_model.with_reward:
            next_obs = obs + target[..., :-1]
            reward = target[..., -1:]
        else:
            next_obs = obs + target
            reward = np.zeros((obs.shape[0], 1))
        terminal = self.terminal_fn(obs, act, next_obs)

        return next_obs, reward, terminal, info


    def set_for_eval_value_gap(self, real_env, eval_policy_set, training_dataset=None):
        self.real_env = real_env
        self.real_env.reset()
        self.eval_policy_set = eval_policy_set

        if training_dataset is not None:
            next_obss = training_dataset["next_observations"]

            max_next_obss = np.max(next_obss, axis=0)
            self.next_obss_pred_upper = max_next_obss + 0.1 * np.abs(max_next_obss)
            min_next_obss = np.min(next_obss, axis=0)
            self.next_obss_pred_lower = min_next_obss - 0.1 * np.abs(min_next_obss)

            rewards = training_dataset["rewards"]

            max_rewards = np.max(rewards)
            min_rewards = np.min(rewards)
            self.reward_pred_upper = max_rewards + 0.1 * np.abs(max_rewards)
            self.reward_pred_lower = min_rewards - 0.1 * np.abs(min_rewards)

    def eval_real_values(self, gamma=0.995, num_eval_episodes=10, deterministic_policy=False, debug=False):
        real_values = []
        reward_means = []
        reward_mins = []
        reward_quantile_1 = []
        reward_quantile_2 = []
        reward_maxs = []
        lengths = []

        self.energy_model.eval()
        
        for idx, policy in enumerate(self.eval_policy_set):
            print(f"eval_idx {idx}")
            rewards = []
            eps_length = []
            value_estimates = []
            obs = self.real_env.reset()
            num_episodes = 0
            value, episode_length = 0, 0
            
            while num_episodes < num_eval_episodes:
                action = policy.select_action(obs, deterministic=deterministic_policy)
                next_obs, reward, terminal, _ = self.real_env.step(action.flatten())
                value += reward * (gamma ** episode_length)
                rewards.append(reward)

                episode_length += 1
                obs = next_obs.copy()

                if terminal:
                    obs = self.real_env.reset()
                    value_estimates.append(value)
                    num_episodes += 1
                    eps_length.append(episode_length)
                    value, episode_length = 0, 0
            real_values.append(np.mean(value_estimates))

            reward_means.append(np.mean(rewards))
            reward_mins.append(np.min(rewards))
            reward_quantile_1.append(np.quantile(rewards, 0.55))
            reward_quantile_2.append(np.quantile(rewards, 0.7))
            reward_maxs.append(np.max(rewards))
            lengths.append(np.mean(eps_length))

        if debug:
            return np.array(real_values), np.array(reward_means), \
                np.array(reward_mins), np.array(reward_quantile_1), np.array(reward_quantile_2), np.array(reward_maxs), \
                np.array(lengths)
            
        return np.array(real_values)


    def eval_fake_values(self, reward_dynamics:EnsembleDynamics=None, init_states=None, gamma=0.995, num_eval_episodes=10, max_step=1000, deterministic_policy=True):
        num_policy = len(self.eval_policy_set)
        traj_dones = np.zeros((num_eval_episodes * num_policy, max_step))
        rewards = np.zeros((num_eval_episodes * num_policy, max_step))

        if init_states is None:
            init_states = [self.real_env.reset() for i in range(num_eval_episodes * num_policy)]
        
        obs = torch.as_tensor(np.array(init_states), dtype=torch.float32)
        if "Ant" in self.real_env.__str__():
            obs = obs[:, :27]

        target_shape = list(obs.shape)
        if self.energy_model.with_reward:
            target_shape[1] += 1

        values = np.zeros((num_eval_episodes * num_policy))

        print("eval idx 0~{} parallelly".format(num_policy-1))

        for step in range(max_step):
            policy_obs = obs
            if "Ant" in self.real_env.__str__():
                policy_obs = np.concatenate([obs, np.zeros((obs.shape[0], 84))], axis=-1)
            act = torch.as_tensor(np.array([self.eval_policy_set[i].select_action(policy_obs[i * num_eval_episodes : (i+1) * num_eval_episodes], deterministic=deterministic_policy) for i in range(num_policy)])).reshape(num_policy * num_eval_episodes, -1)
            obs_act = torch.cat([obs, act], dim=-1)
            obs_act = self.scaler.transform(obs_act)

            init_target = torch.zeros(target_shape, device=self.device)
            target, info = self.predict(obs_act, init_target)
            # next_obs = obs + target[..., :-1]
            # reward = target[..., -1:]
            if self.energy_model.with_reward == True:
                next_obs = obs + target[..., :-1]
            else:
                next_obs = obs + target

            if self.next_obss_pred_lower is not None and self.next_obss_pred_upper is not None:
                next_obs = np.clip(next_obs, self.next_obss_pred_lower, self.next_obss_pred_upper)

            if reward_dynamics is None:
                reward = target[..., -1]
            else: 
                _, reward, _, _ = reward_dynamics.step(obs.cpu().numpy(), act.cpu().numpy())

            if self.reward_pred_lower is not None and self.reward_pred_upper is not None:
                reward = np.clip(reward, self.reward_pred_lower, self.reward_pred_upper) 
            terminal = self.terminal_fn(obs.cpu().numpy(), act.cpu().numpy(), next_obs.cpu().numpy()).squeeze()

            traj_dones[:, step] = np.where(terminal, terminal, traj_dones[:, step-1] if step > 0 else np.zeros((num_eval_episodes * num_policy,)))  ## 

            if all(traj_dones[:, step] == 1):
                break

            values = values + reward.squeeze() * (gamma ** step) * (1 - traj_dones[:,step])  ## 
            rewards[traj_dones[:,step] == 0, step] = reward[traj_dones[:,step] == 0].squeeze()

            obs = next_obs

        policy_values = values.reshape(num_policy, num_eval_episodes).mean(axis=-1)

        return policy_values
    
    
    def learn_energy(
            self,
            data: Dict,
            logger: Logger,
            max_epochs: Optional[float] = None,
            batch_size: int = 256,
            eval_holdout: bool = True,
            eval_ratio: float = 0.1,
            max_size: int = 1000
            ):
        states_actions, raw_delta_states = self.format_samples_for_energy_training(data, with_reward=self.energy_model.with_reward)
        data_size = states_actions.shape[0]

        eval_size = max(int(np.ceil(eval_ratio * data_size)), max_size)
        if eval_holdout:
            train_size = data_size - eval_size
            train_splits, eval_splits = torch.utils.data.random_split(range(data_size), (train_size, eval_size))
            train_states_actions, train_delta_states = states_actions[train_splits], raw_delta_states[train_splits]
            eval_states_actions, eval_raw_delta_states = states_actions[eval_splits], raw_delta_states[eval_splits]
        else:
            train_size = data_size
            train_states_actions, train_delta_states = states_actions, raw_delta_states
            eval_indices = np.random.choice(data_size, eval_size)
            eval_states_actions, eval_raw_delta_states = states_actions[eval_indices], raw_delta_states[eval_indices]

        self.scaler.fit(train_states_actions)
        self.delta_scaler.fit(train_delta_states)

        print("delta_state: mu={}, std={}".format(self.delta_scaler.mu, self.delta_scaler.std))

        train_states_actions = self.scaler.transform(train_states_actions)
        train_delta_states = self.delta_scaler.transform(train_delta_states)
        eval_states_actions = self.scaler.transform(eval_states_actions)


        for epoch in range(max_epochs):
            self.energy_model.train()

            train_idxes = np.random.choice(np.arange(train_size), train_size, replace=False)


            train_states_actions = train_states_actions[train_idxes]
            train_delta_states = train_delta_states[train_idxes]


            for batch_num in range(int(np.ceil(train_size / batch_size))):

                train_states_actions_batch = train_states_actions[batch_num * batch_size : (batch_num + 1) * batch_size]

                train_delta_states_batch = train_delta_states[batch_num * batch_size : (batch_num + 1) * batch_size]

                loss_dict, time_info = self.energy_learn_step(train_states_actions_batch, train_delta_states_batch)

                logger.logkv_mean("loss/etm_total_loss", loss_dict["etm_total_loss"])

                if "grad_loss" in loss_dict.keys():
                    logger.logkv_mean("loss/grad_loss", loss_dict["grad_loss"])
                    logger.logkv_mean("loss/contrastive_loss", loss_dict["contrastive_loss"])
                logger.logkv_mean("time/neg_sample_time", time_info["neg_sample_time"])
                logger.logkv_mean("time/etm_loss_time", time_info["etm_loss_time"])
                logger.logkv_mean("time/grad_loss_time", time_info["grad_loss_time"])
                logger.logkv_mean("time/backward_time", time_info["backward_time"])
                logger.logkv_mean("loss/grad_norm_mean", loss_dict["grad_norm_mean"])
            
            eval_loss, eval_info = self.evaluate(eval_states_actions, eval_raw_delta_states)

            logger.logkv("eval/error", eval_loss)
            logger.logkv("eval/mcmc_time", eval_info["mcmc_time_sum"])

            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_process"])
            if epoch % 10 == 0:
                ckpt_dir = os.path.join(logger.checkpoint_dir, "epoch_{}".format(epoch))
                os.makedirs(ckpt_dir)
                self.save(ckpt_dir)

        self.save(logger.model_dir)


    def energy_learn_step(
            self,
            states_actions,
            delta_states,
            ):
        start_time = time.time()

        batch_size = states_actions.shape[0]
        states_actions = torch.as_tensor(states_actions, dtype=torch.float32, device=self.energy_model.device)
        expanded_delta_states = torch.as_tensor(delta_states[:, None, :], dtype=torch.float32, device=self.energy_model.device)
        repeated_states_actions = torch.repeat_interleave(states_actions, self.num_negative_samples + 1, dim=0)

        neg_delta_states, combined_pos_neg_delta_states = self.create_negative_samples(
            states_actions, 
            expanded_delta_states, 
            batch_size,
            )
        neg_sample_time = time.time()

        predictions = self.energy_model(repeated_states_actions, combined_pos_neg_delta_states)
        predictions = predictions.reshape(batch_size, self.num_negative_samples + 1)

        per_sample_loss = self.etm_loss(batch_size, predictions)
        contrastive_loss = per_sample_loss.sum().item()
        etm_loss_time = time.time()

        grad_loss, grad_norm_mean = self.grad_penalty(batch_size, repeated_states_actions, combined_pos_neg_delta_states, grad_margin=self.grad_penalty_margin, create_graph=self.add_grad_penalty)
        grad_loss_time = time.time()

        if self.add_grad_penalty:
            per_sample_loss = per_sample_loss + grad_loss

        total_loss = per_sample_loss.sum()

        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()
        backward_time = time.time()

        losses_dict = {
            'etm_total_loss': total_loss.item(),
            'grad_norm_mean': grad_norm_mean,
        }
        if grad_loss is not None:
            losses_dict['contrastive_loss'] = contrastive_loss
            losses_dict['grad_loss'] = grad_loss.sum().item()
            losses_dict['grad_loss_mean'] = grad_loss.mean().item()

        return losses_dict, {'neg_sample_time': neg_sample_time - start_time,
                             'etm_loss_time': etm_loss_time - neg_sample_time,
                             'grad_loss_time': grad_loss_time - etm_loss_time,
                             'backward_time': backward_time - grad_loss_time}
    
    def etm_loss(
            self, 
            batch_size, # B
            predictions, # [B x n+1] with true in column [:, -1]
            ):
        
        if self.etm_loss_type == 'info_nce':
            def info_nce(predictions, batch_size, num_negative_samples, softmax_temperature):
                logsoftmax_predictions = - predictions / softmax_temperature - torch.logsumexp( - predictions / softmax_temperature, dim=-1, keepdim=True)
                indices = torch.ones((batch_size,), dtype=torch.int64, device=logsoftmax_predictions.device) * num_negative_samples
                labels = F.one_hot(indices, num_classes=num_negative_samples+1).float()
                per_example_loss = torch.nn.KLDivLoss(reduction="none")(logsoftmax_predictions, labels).sum(dim=-1)
                return per_example_loss
            per_example_loss = info_nce(predictions, batch_size, self.num_negative_samples, self.softmax_temperature)

        return per_example_loss
    
    def grad_penalty(
            self, 
            batch_size, 
            states_actions,
            combined_pos_neg_samples,

            grad_margin: float=1.0,
            square_grad_penalty: bool=True,
            create_graph: bool=True,
            ):
        grad, _ = grad_wrt_next_s(self.energy_model, states_actions, combined_pos_neg_samples.detach(), create_graph=create_graph)

        grad_norms = torch.norm(grad, dim=-1).view((batch_size, -1))
        grad_norm_mean = grad_norms.mean().item()

        if grad_margin is not None:
            grad_norms = grad_norms - grad_margin

            grad_norms = torch.clamp(grad_norms, 0., 1e10)

        if square_grad_penalty:
            grad_norms = grad_norms ** 2

        grad_loss = torch.mean(grad_norms, dim=1)

        return grad_loss, grad_norm_mean

    def create_negative_samples(
            self, 
            states_actions, # B x state_act_spec
            expanded_delta_states, # B x 1 x state_spec
            batch_size,
            ):

        negative_samples_delta_states = []

        repeated_states_actions = torch.repeat_interleave(states_actions, self.num_langevin_samples, dim=0)
        random_delta_state_samples = 2 * torch.rand(batch_size * self.num_langevin_samples, expanded_delta_states.shape[-1], device=self.energy_model.device) - 1
        langevin_delta_states = langevin_mcmc_sa_s(self.energy_model, 
                                                    repeated_states_actions, 
                                                    random_delta_state_samples, 
                                                    num_iterations = self.langevin_iterations, 
                                                    noise_scale = self.langevin_noise_scale, 
                                                    grad_clip = self.langevin_grad_clip, 
                                                    delta_clip = self.langevin_delta_clip, 
                                                    margin_clip = 1.1,
                                                    sampler_stepsize_init = self.langevin_stepsize_init,
                                                    sampler_stepsize_final = self.langevin_stepsize_final,
                                                    )
        negative_samples_delta_states.append(langevin_delta_states.reshape(batch_size, self.num_langevin_samples, -1))

        neg_samples = torch.cat(negative_samples_delta_states, dim=1)
        combined_pos_neg_samples = torch.cat([neg_samples, expanded_delta_states], dim=1).reshape(-1, expanded_delta_states.shape[-1])
        # B*(num_negative_samples + 1) x state_spec

        return neg_samples, combined_pos_neg_samples
    
    def predict(self, state_action, init_samples, langevin_steps=100, langevin_noise=0.0, rescale: bool = True, return_tensor=False):

        mcmc_start_time = time.time()
        delta_state = langevin_mcmc_sa_s(self.energy_model, state_action, init_samples, langevin_steps, noise_scale=langevin_noise, grad_clip=0.5, delta_clip=0.5, margin_clip=1.1)
        device = delta_state.device
        delta_state = delta_state.detach().cpu().numpy()
        mcmc_time = time.time() - mcmc_start_time

        info = {
            "mcmc_time": mcmc_time
        }
        if rescale is True:
            delta_state = self.delta_scaler.inverse_transform(delta_state)
            
        if return_tensor:
            delta_state = torch.as_tensor(delta_state, dtype=torch.float32, device=device)

        return delta_state, info

    def evaluate(self, states_actions, delta_states, batch_size: int = 51200, error_type="mse"):

        self.energy_model.eval()
        states_actions = torch.as_tensor(states_actions, dtype=torch.float32, device=self.energy_model.device)
        delta_states = torch.as_tensor(delta_states, dtype=torch.float32, device=self.energy_model.device)

        samples_size = states_actions.shape[0]
        error_sum = 0
        mcmc_time = 0
        
        for batch_num in range(int(np.ceil(samples_size / batch_size))):
            states_actions_batch = states_actions[batch_num * batch_size : (batch_num + 1) * batch_size]
            delta_states_batch = delta_states[batch_num * batch_size : (batch_num + 1) * batch_size]

            pred_delta_states, pred_info = self.predict(states_actions_batch, init_samples=torch.zeros_like(delta_states_batch, device=self.energy_model.device), rescale=True)
            
            if error_type == "mse":
                error_sum += np.sum((pred_delta_states - delta_states_batch.cpu().numpy()) ** 2).item()
            elif error_type == "mae":
                error_sum += np.sum(np.abs(pred_delta_states - delta_states_batch.cpu().numpy())).item()
            mcmc_time += pred_info["mcmc_time"]

        error_mean = error_sum / samples_size / (delta_states.shape[-1]+1)
        return error_mean, {"mcmc_time_sum": mcmc_time}
    

    def format_samples_for_energy_training(self, data: Dict, with_reward: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        delta_obss = next_obss - obss

        if with_reward:
            rewards = data["rewards"]

            if len(rewards.shape) == 1:
                rewards = rewards.reshape(-1, 1)
            delta_obss = np.concatenate((delta_obss, rewards), axis=-1)
            print(delta_obss.shape)

        obss_actions = np.concatenate((obss, actions), axis=-1)
        return obss_actions, delta_obss

    
    def save(self, save_path: str):
        torch.save(self.energy_model.state_dict(), os.path.join(save_path, "energy.pth"))
        self.scaler.save_scaler(save_path)
        self.delta_scaler.save_scaler(save_path, 'delta_')

    def load(self, load_path: str):
        self.energy_model.load_state_dict(torch.load(os.path.join(load_path, "energy.pth"), map_location=self.energy_model.device))
        self.scaler.load_scaler(load_path)
        self.delta_scaler.load_scaler(load_path, 'delta_')

    
class EnsembleEnergyDynamics():

    def __init__(
        self,
        energy_dynamics_list: List[EnergyDynamics],
        penalty_coef: float = 0,
        uncertainty_mode: str = "ensemble_std",
    ) -> None:
        
        self.dynamics_list = energy_dynamics_list
        self.num_ensemble = len(self.dynamics_list)
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

        print("Ensemble {} Energy Dynamics".format(self.num_ensemble))

    def step(
            self,
            obs: np.ndarray,
            action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        batch_size = obs.shape[0]
        model_idxs = np.random.choice(self.num_ensemble, batch_size)
        next_obss, rewards, terminals = [], [], []
        for i, dynamics in enumerate(self.dynamics_list):
            next_obs, reward, terminal, step_info = dynamics.step(obs, action)
            next_obss.append(next_obs)
            rewards.append(reward)
            terminals.append(terminal)
        ensemble_next_obs = np.array(next_obss)
        ensemble_reward = np.array(rewards)
        ensemble_terminal = np.array(terminals)

        selected_next_obs = ensemble_next_obs[model_idxs, np.arange(batch_size)]
        selected_reward = ensemble_reward[model_idxs, np.arange(batch_size)]
        selected_terminal = ensemble_terminal[model_idxs, np.arange(batch_size)]

        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef > 0:
            if self._uncertainty_mode == "ensemble_std":
                penalty = np.linalg.norm(ensemble_next_obs.std(axis=0), axis=-1, keepdims=True)
                info["penalty"] = penalty


        return selected_next_obs, selected_reward, selected_terminal, info

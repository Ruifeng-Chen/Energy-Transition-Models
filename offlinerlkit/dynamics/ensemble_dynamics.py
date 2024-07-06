import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.reward_only = self.model._reward_only

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        specified_idx = None,
        deterministic = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        if not self.reward_only:
            mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std * (1-deterministic)).astype(np.float32)

        num_models, batch_size, _ = ensemble_samples.shape
        if specified_idx is None:
        # choose one model from ensemble
            model_idxs = self.model.random_elite_idxs(batch_size)
        elif isinstance(specified_idx, int):
            model_idxs = np.ones((batch_size, ), dtype=int) * specified_idx
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs, terminal = None, None
        if not self.reward_only:
            next_obs = samples[..., :-1]
            terminal = self.terminal_fn(obs, action, next_obs)

        reward = samples[..., -1:]
        reward = reward * self.reward_scale + self.reward_bias

        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        if self.reward_only:
            targets = rewards.reshape(-1, 1)
        else:
            targets = np.concatenate((delta_obss, rewards.reshape(-1, 1)), axis=-1)
        return inputs, targets

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout: bool = True,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
        min_epoch: int = None,
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        if holdout:
            train_size = data_size - holdout_size
            train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
            train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
            holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]
        else:
            train_size = data_size
            train_inputs, train_targets = inputs, targets
            holdout_indices = np.random.choice(data_size, holdout_size)
            holdout_inputs, holdout_targets = inputs[holdout_indices], targets[holdout_indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                if not (min_epoch is not None and epoch <= min_epoch):
                    break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    @ torch.no_grad()
    def evaluate(self, inputs: np.ndarray, targets: np.ndarray, batch_size=51200) -> List[float]:
        self.model.eval()
        samples_size = inputs.shape[0]
        error_sum = 0
        for batch_num in range(int(np.ceil(samples_size / batch_size))):
            inputs_batch = inputs[batch_num * batch_size : (batch_num + 1) * batch_size]
            targets_batch = targets[batch_num * batch_size : (batch_num + 1) * batch_size]
            # targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            mean, _ = self.model(inputs_batch)
            error_sum += (np.abs(mean.cpu().numpy() - targets_batch)).sum(axis=(1, 2))
        # loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        print(samples_size)
        print(inputs.shape[-1])
        val_loss = list(error_sum/samples_size/inputs.shape[-1])
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)

    def set_for_eval_value_gap(self, real_env, eval_policy_set, training_dataset=None):
        self.real_env = real_env
        self.real_env.reset()
        self.eval_policy_set = eval_policy_set

        if training_dataset is not None:
            next_obss = training_dataset["next_observations"]

            max_next_obss = np.max(next_obss, axis=0)
            self.next_obss_pred_upper = max_next_obss + 0.0 * np.abs(max_next_obss)
            min_next_obss = np.min(next_obss, axis=0)
            self.next_obss_pred_lower = min_next_obss - 0.0 * np.abs(min_next_obss)

            rewards = training_dataset["rewards"]

            max_rewards = np.max(rewards)
            min_rewards = np.min(rewards)
            self.reward_pred_upper = max_rewards + 0.05 * np.abs(max_rewards)
            self.reward_pred_lower = min_rewards - 0.05 * np.abs(min_rewards)

    def eval_real_values(self, gamma=0.995, num_eval_episodes=10, deterministic_policy=False):
        real_values = []
        self.model.eval()
        
        for idx, policy in enumerate(self.eval_policy_set):
            print(f"eval_idx {idx}")
            value_estimates = []
            obs = self.real_env.reset()
            num_episodes = 0
            value, episode_length = 0, 0
            
            while num_episodes < num_eval_episodes:
                action = policy.select_action(obs, deterministic=deterministic_policy)
                next_obs, reward, terminal, _ = self.real_env.step(action.flatten())
                value += reward * (gamma ** episode_length)
                episode_length += 1
                obs = next_obs.copy()

                if terminal:
                    obs = self.real_env.reset()
                    value_estimates.append(value)
                    num_episodes += 1
                    value, episode_length = 0, 0
            real_values.append(np.mean(value_estimates))
            
        return np.array(real_values)

    def eval_fake_values(self, gamma=0.995, num_eval_episodes=10, max_step=1000, model_idx=None, deterministic_policy=False):
        fake_values = []
        self.model.eval()

        for idx, policy in enumerate(self.eval_policy_set):
            print(f"eval_idx {idx}")

            fake_ep_info = []
            obs = self.real_env.reset()
            obs_dim = obs.shape[-1]
            # step = 0
            num_episodes = 0
            value, episode_length = 0, 0

            while num_episodes < num_eval_episodes:
                obs = obs.reshape(-1, obs_dim)
                action = policy.select_action(obs, deterministic=deterministic_policy)
                next_obs, reward, terminal, _ = self.step(obs, action, specified_idx=model_idx)
                next_obs = np.clip(next_obs, self.next_obss_pred_lower, self.next_obss_pred_upper)
                reward, terminal = reward.flatten()[0], terminal.flatten()[0]
                reward = np.clip(reward, self.reward_pred_lower, self.reward_pred_upper)
                value += reward * (gamma ** episode_length)
                episode_length += 1
                # step += 1
                obs = next_obs.copy()
                if terminal or episode_length >= max_step:
                    fake_ep_info.append(
                        {"value": value, "episode_length": episode_length}
                    )
                    # step = 0
                    num_episodes += 1
                    value, episode_length = 0, 0
                    obs = self.real_env.reset()

            fake_values.append(np.mean([info["value"] for info in fake_ep_info]))

        return np.array(fake_values)
    
    def collect_real_traj(self, gamma=0.995, num_eval_episodes=1, real_actions=None):
        real_states = {}
        real_actions = {}
        real_length = {}
        self.model.eval()
        
        for idx, policy in enumerate(self.eval_policy_set):
            print(f"eval_idx {idx}")
            value_estimates = []
            obs = self.real_env.reset()
            num_episodes = 0
            value, episode_length = 0, 0

            real_states[idx] = []
            real_actions[idx] = []
            real_length[idx] = 0
            
            while num_episodes < num_eval_episodes:
                action = policy.select_action(obs, deterministic=True)
                next_obs, reward, terminal, _ = self.real_env.step(action.flatten())

                value += reward * (gamma ** episode_length)
                episode_length += 1

                img = self.real_env.render(mode="rgb_array")
                real_states[idx].append(img)
                real_actions[idx].append(action)
                real_length[idx] = episode_length

                obs = next_obs.copy()

                if terminal:
                    obs = self.real_env.reset()
                    value_estimates.append(value)
                    num_episodes += 1
                    # value, episode_length = 0, 0

            print(f"eval_idx {idx} length {episode_length}")

        return {"obs": real_states, "act": real_actions, "len": real_length}

    def collect_fake_traj(self, gamma=0.995, num_eval_episodes=1, max_step=1000, model_idx=None, real_actions=None):
        fake_states = {}
        fake_length = {}
        self.model.eval()

        for idx, policy in enumerate(self.eval_policy_set):
            print(f"eval_idx {idx}")

            fake_ep_info = []
            obs = self.real_env.reset()
            obs_dim = obs.shape[-1]
            # step = 0
            num_episodes = 0
            value, episode_length = 0, 0

            fake_states[idx] = []
            fake_length[idx] = 0

            # while num_episodes < num_eval_episodes:
            while episode_length < max_step:
                obs = obs.reshape(-1, obs_dim)
                if real_actions is not None and episode_length < len(real_actions[idx]):
                    action = real_actions[idx][episode_length]
                else:
                    action = policy.select_action(obs, deterministic=True)

                next_obs, reward, terminal, _ = self.step(obs, action, specified_idx=model_idx)
                reward, terminal = reward.flatten()[0], terminal.flatten()[0]
                value += reward * (gamma ** episode_length)

                try:
                    qpos, qvel = obs[0, :obs_dim//2], obs[0, obs_dim//2:]
                    self.real_env.set_state(np.concatenate([[0], qpos]), qvel)
                    img = self.real_env.render(mode="rgb_array")
                    fake_states[idx].append(img)
                    fake_length[idx] = episode_length
                except:
                    print("error at step {}".format(episode_length))

                episode_length += 1

                # step += 1
                obs = next_obs.copy()
                # if terminal or episode_length >= max_step:
                #     # step = 0
                #     num_episodes += 1
                #     # value, episode_length = 0, 0
                #     obs = self.real_env.reset()

            print(f"eval_idx {idx} length {episode_length}")

        return {"obs": fake_states, "act": real_actions, "len": fake_length}
    
    def eval(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout: bool = True,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
        mesh_grid=True,
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        
        raw_inputs = inputs
        
        inputs = self.scaler.transform(inputs)
        new_holdout_losses, info = self.validate_samples(inputs, targets)
        
        print(raw_inputs.shape)
        print(raw_inputs.reshape(21, 1001, -1)[0, :, 0])
        print(raw_inputs.reshape(21, 1001, -1)[0, :, 1])
        
        print(info["sample"].reshape(21, 1001, -1)[0, :, 0])
        print(info["sample"].reshape(21, 1001, -1)[0, :, 1])
        assert 0
        
        resss = info["sample"][:, :] + raw_inputs[:, :-1]
        print("save_fwm")
        print(resss.shape)
        np.save("data_fwm.npy", resss)
        
        return new_holdout_losses
    
    @ torch.no_grad()
    def validate_samples(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss, {"sample": mean[0, :, :-1].cpu().numpy()}
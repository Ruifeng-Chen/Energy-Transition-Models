import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple, List
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import EnergyDynamics, EnsembleEnergyDynamics


class EMPOPolicy(SACPolicy):

    def __init__(
        self,
        dynamics: EnsembleEnergyDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_type: str = "ensemble_std",
        penalty_coef: float = 0,
        penalty_decay: bool = False,
        deterministic_backup: bool = False,
        target_zero_clip: bool = False,
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            deterministic_backup=deterministic_backup,
        )

        self.dynamics = dynamics
        
        self._penalty_coef = penalty_coef
        self.penalty_type = penalty_type
        self.penalty_decay = penalty_decay

        self._target_zero_clip = target_zero_clip

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)  
            rollout_transitions["terminals"].append(terminals)

            if self.penalty_type == "ensemble_std":
                rollout_transitions["penalties"].append(info["penalty"])


            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        rollout_info = {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

        if "penalty" in info.keys():
            rollout_info["penalty_mean"] = info["penalty"].mean()

        return rollout_transitions, rollout_info

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        
        obss, actions, next_obss, terminals = mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], mix_batch["terminals"]
        real_size = real_batch["observations"].shape[0]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            )

            if not self._deterministic_backup:
                next_q -= self._alpha * next_log_probs

            penalty_coef = self._penalty_coef
            if self.penalty_type == "ensemble_std":
                penalty = fake_batch["penalties"]
                if self.penalty_decay:
                    penalty_coef = penalty_coef * torch.clamp(next_q[real_size:] / 100, 0.0, 1.0)
                fake_rewards = fake_batch["rewards"] - penalty_coef * penalty
                rewards = torch.cat([real_batch["rewards"], fake_rewards], 0)

            target_q = rewards + self._gamma * (1 - terminals) * next_q

            if self._target_zero_clip:
                raw_target_mean = target_q.mean()
                target_q = torch.clamp(target_q, 0, None)

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "critic/q1_mean": q1.mean().item(),
            "critic/q2_mean": q2.mean().item(),
            "actor/log_prob": next_log_probs.mean().item(),
            "reward_mean": rewards.mean().item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        if "raw_target_mean" in locals().keys():
            result["raw_target_mean"] = raw_target_mean.item()
        if "penalty" in locals().keys():
            result["penalty_mean"] = penalty.mean().item()
            result["penalized_reward_mean"] = fake_rewards.mean().item()
        if self.penalty_decay:
            result["penlaty_coef_mean"] = penalty_coef.mean().item()
            result["penlaty_coef_max"] = penalty_coef.max().item()
            result["penlaty_coef_min"] = penalty_coef.min().item()

        return result

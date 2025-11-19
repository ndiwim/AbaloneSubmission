from typing import Optional, Union

import random
import helper as h
import gymenv.moveGenerator as mg

import numpy as np
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise



class MaskedDQN(DQN):
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
            replacing the root sample action, such that I can get 
        """
        # Select action randomly or according to policy
        assert self._last_obs is not None, "self._last_obs was not set"
        
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.zeros(self._last_obs.shape[0], dtype=np.int16)
            for obs_index in range(self._last_obs.shape[0]):
                obs = self._last_obs[obs_index].argmax(axis=0)
                unscaled_action[obs_index] = random.choice(mg.generate_valid_moves(0, obs))
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([random.choice(mg.generate_valid_moves(0, observation[batch_no].argmax(axis=0))) for batch_no in range(n_batch)])
            else:
                action = np.array(random.choice(mg.generate_valid_moves(0, observation.argmax(axis=0))))
        else:
            action, state = self.policy.predict(observation, state, episode_start, True)
        return action, state
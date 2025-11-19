import torch    
import random
import numpy as np    
import helper as h
import gymenv.moveGenerator as mg
from torch.distributions import Categorical
from stable_baselines3 import DQN, PPO, A2C
from models.maskedDQN import MaskedDQN
from stable_baselines3.a2c.policies import MlpPolicy, CnnPolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CnnMlpExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, action_space=None, is_dqn=False):
        if action_space is None:
            raise RuntimeError("need an actions space in policy kwargs")
        
        if hasattr(action_space, 'n'): 
            mlp_dim = 256 if action_space.n < 1000 else 1024 
            features_dim = action_space.n
        else: 
            mlp_dim = 256 if action_space.nvec[0] == 9 else 1024 
            features_dim = mlp_dim
        
        self.is_dqn = is_dqn
        super().__init__(observation_space, features_dim)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, features_dim),
        )
        
        self.to(h.device)

    def forward(self, obs):
        obs = obs.to(h.device)
        return self.linear(self.cnn(obs))


class MaskedPolicyBased(CnnPolicy):
    def __init__(self, *args, agent_player=h.Cells.PlayerOne, enable_rotation=True, raise_error=True, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn = 0 if enable_rotation or agent_player == h.Cells.PlayerOne else 1
        self.raise_error = raise_error
    
    def forward(self, obs, deterministic=False):
        if not self.raise_error:
            actions, values, log_prob = super().forward(obs, deterministic)
            return actions, values, log_prob
            
        # Extract features and policy/value latents
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)   # shape: [batch_size, n_actions]

        batch_size = logits.shape[0]

        # === MASKING ===
        # Create a mask for each observation in the batch
        masks = []
        for i in range(batch_size):
            single_obs = obs[i]
            as_normal_grid = single_obs.argmax(dim=0)
            valid_move_indices = mg.generate_valid_moves(self.turn, as_normal_grid) 
            mask = np.isin(np.arange(single_obs.shape[-1] ** 4 * 6), valid_move_indices)
            
            masks.append(mask)

        masks = [torch.as_tensor(m, dtype=torch.float32, device=logits.device) for m in masks]
        masks = torch.stack(masks)  # shape [batch_size, n_actions]

        # Apply mask: invalid actions -> -inf
        logits[masks == 0] = float('-inf')

        # --- Construct distribution with masked logits ---
        masked_logits = logits  # shape [batch_size, n_actions]
        dist = Categorical(logits=masked_logits)

        if deterministic:
            actions = torch.argmax(masked_logits, dim=-1)
        else:
            actions = dist.sample()
        
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf).flatten()
        
        return actions, values, log_prob
    
    def _predict(self, observation, deterministic=False):
        if not self.raise_error:
            return super()._predict(observation, deterministic)
        
        obs = observation.unsqueeze(0) if len(observation.shape) == 1 else observation
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions.squeeze(0)
    

class MDPolicyBasedIndependent(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_heads = torch.nn.ModuleList([
            torch.nn.Linear(self.features_dim, n) for n in self.action_space.nvec
        ])
    
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        values = self.value_net(features)

        log_prob = torch.zeros(obs.shape[0], device=obs.device)
        actions = []
        for head_index in range(len(self.action_heads)):
            logits = self.action_heads[head_index](features)
            dist = Categorical(logits=logits)
            
            if deterministic:
                action = dist.logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob += dist.log_prob(action)
            actions.append(action)
        
        actions = torch.stack(actions, dim=-1)

        return actions, values, log_prob
    
    def _predict(self, observation, deterministic=False):
        obs = observation.unsqueeze(0) if len(observation.shape) == 1 else observation
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions.squeeze(0)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.extract_features(obs)
        values = self.value_net(features)

        log_probs = []
        entropies = []

        for i, head in enumerate(self.action_heads):
            logits = head(features)
            dist = Categorical(logits=logits)

            # actions[:, i] corresponds to that head’s action
            head_actions = actions[:, i]

            log_probs.append(dist.log_prob(head_actions))
            entropies.append(dist.entropy())

        # Combine
        log_prob = torch.stack(log_probs, dim=1).sum(dim=1)      # [batch_size]
        entropy = torch.stack(entropies, dim=1).sum(dim=1)      # [batch_size]

        return values, log_prob, entropy


class MDPolicyBasedDependent(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_size = 256
        
        self.layerNorms = torch.nn.ModuleList()
        self.embeddings = torch.nn.ModuleList()
        
        self.action_heads = torch.nn.ModuleList()
        for index, action_space in enumerate(self.action_space.nvec):
            self.action_heads.append(torch.nn.Linear(self.features_dim + self.embed_size * index, action_space))
            
            if index < len(self.action_space.nvec) - 1:
                self.embeddings.append(torch.nn.Embedding(action_space, self.embed_size))
                self.layerNorms.append(torch.nn.LayerNorm(self.features_dim + self.embed_size * (index + 1)))
    
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        values = self.value_net(features)

        embeddings = torch.zeros((obs.shape[0], self.embed_size * (len(self.action_space.nvec) - 1)), device=obs.device)
        log_prob = torch.zeros(obs.shape[0], device=obs.device)
        actions = torch.zeros((obs.shape[0], len(self.action_space.nvec)), device=obs.device, dtype=torch.long)
        
        for head_index, head in enumerate(self.action_heads):
            if head_index == 0:
                logits = head(features)
            else:
                embeddings[
                    :, (head_index - 1) * self.embed_size: head_index * self.embed_size
                ] = self.embeddings[head_index - 1](actions[:, head_index - 1])
                
                new_features = torch.cat([features, embeddings[:, :head_index * self.embed_size]], dim=-1)
                normalized_new_features = self.layerNorms[head_index - 1](new_features)
                
                logits = head(normalized_new_features)
                
            dist = Categorical(logits=logits)
                
            action = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
            actions[:, head_index] = action
            
            log_prob += dist.log_prob(action)
        
        return actions, values, log_prob
    
    def _predict(self, observation, deterministic=False):
        obs = observation.unsqueeze(0) if len(observation.shape) == 1 else observation
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions.squeeze(0)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        actions_long = actions.to(torch.long)
        
        features = self.extract_features(obs)
        values = self.value_net(features)

        batch_size = obs.shape[0]
        
        embeddings = torch.zeros((batch_size, (len(self.action_space.nvec) - 1) * self.embed_size), device=obs.device)
        log_prob = torch.zeros(batch_size, device=obs.device)
        entropy = torch.zeros(batch_size, device=obs.device)

        for head_index, head in enumerate(self.action_heads):
            if head_index == 0:
                logits = head(features)
            else:
                embeddings[
                    :, (head_index - 1) * self.embed_size: head_index * self.embed_size
                ] = self.embeddings[head_index - 1](actions_long[:, head_index - 1])
                
                new_features = torch.cat([features, embeddings[:, :head_index * self.embed_size]], dim=-1)
                normalized_new_features = self.layerNorms[head_index - 1](new_features)
                
                logits = head(normalized_new_features)
            
            
            dist = Categorical(logits=logits)

            log_prob += dist.log_prob(actions[:, head_index])
            entropy += dist.entropy()

        return values, log_prob, entropy


class MDPolicyBasedMasked(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_size = 256
        
        self.layerNorms = torch.nn.ModuleList()
        self.embeddings = torch.nn.ModuleList()
        
        self.action_heads = torch.nn.ModuleList()
        for index, action_space in enumerate(self.action_space.nvec):
            self.action_heads.append(torch.nn.Linear(self.features_dim + self.embed_size * index, action_space))
            
            if index < len(self.action_space.nvec) - 1:
                self.embeddings.append(torch.nn.Embedding(action_space, self.embed_size))
                self.layerNorms.append(torch.nn.LayerNorm(self.features_dim + self.embed_size * (index + 1)))
    
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        values = self.value_net(features)

        embeddings = torch.zeros((obs.shape[0], self.embed_size * (len(self.action_space.nvec) - 1)), device=obs.device)
        log_prob = torch.zeros(obs.shape[0], device=obs.device)
        actions = torch.zeros((obs.shape[0], len(self.action_space.nvec)), device=obs.device, dtype=torch.long)
        traversable_valid_actions = mg.generate_valid_moves_batch_md(obs)
        
        for head_index, head in enumerate(self.action_heads):
            if head_index == 0:
                logits = head(features)
            else:
                embeddings[
                    :, (head_index - 1) * self.embed_size: head_index * self.embed_size
                ] = self.embeddings[head_index - 1](actions[:, head_index - 1])
                
                new_features = torch.cat([features, embeddings[:, :head_index * self.embed_size]], dim=-1)
                normalized_new_features = self.layerNorms[head_index - 1](new_features)
                
                logits = head(normalized_new_features)
            
            # Masking mental math here
            for batch_index in range(obs.shape[0]):
                if head_index == 0:
                    valid_indices = list(traversable_valid_actions[batch_index].keys())
                elif head_index == 1:
                    start_cell_index = actions[batch_index, 0].item()
                    valid_indices = list(traversable_valid_actions[batch_index][start_cell_index].keys())
                else:
                    start_cell_index = actions[batch_index, 0].item()
                    selection_direction_magnitude = actions[batch_index, 1].item()
                    valid_indices = traversable_valid_actions[batch_index][start_cell_index][selection_direction_magnitude]
                    
                mask = np.zeros(self.action_space.nvec[head_index])
                mask[valid_indices] = 1
                
                if mask.sum() == 0:
                    h.print_hex_grid(obs[batch_index, :].numpy().argmax(axis=-0))
                    raise RuntimeError("Got a problem buddy")
                
                logit = logits[batch_index, :]
                logit[mask == 0] = -float('inf')
            
            
            dist = Categorical(logits=logits)
                
            action = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
            actions[:, head_index] = action
            
            log_prob += dist.log_prob(action)
        
        return actions, values, log_prob
    
    def _predict(self, observation, deterministic=False):
        obs = observation.unsqueeze(0) if len(observation.shape) == 1 else observation
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions.squeeze(0)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        actions_long = actions.to(torch.long)
        
        features = self.extract_features(obs)
        values = self.value_net(features)

        batch_size = obs.shape[0]
        
        embeddings = torch.zeros((batch_size, (len(self.action_space.nvec) - 1) * self.embed_size), device=obs.device)
        log_prob = torch.zeros(batch_size, device=obs.device)
        entropy = torch.zeros(batch_size, device=obs.device)
        traversable_valid_actions = mg.generate_valid_moves_batch_md(obs)

        for head_index, head in enumerate(self.action_heads):
            if head_index == 0:
                logits = head(features)
            else:
                embeddings[
                    :, (head_index - 1) * self.embed_size: head_index * self.embed_size
                ] = self.embeddings[head_index - 1](actions_long[:, head_index - 1])
                
                new_features = torch.cat([features, embeddings[:, :head_index * self.embed_size]], dim=-1)
                normalized_new_features = self.layerNorms[head_index - 1](new_features)
                
                logits = head(normalized_new_features)
            
            # Masking mental math here
            for batch_index in range(obs.shape[0]):
                if head_index == 0:
                    valid_indices = list(traversable_valid_actions[batch_index].keys())
                elif head_index == 1:
                    start_cell_index = actions_long[batch_index, 0].item()
                    valid_indices = list(traversable_valid_actions[batch_index][start_cell_index].keys())
                else:
                    start_cell_index = actions_long[batch_index, 0].item()
                    selection_direction_magnitude = actions_long[batch_index, 1].item()
                    valid_indices = traversable_valid_actions[batch_index][start_cell_index][selection_direction_magnitude]
                    
                mask = np.zeros(self.action_space.nvec[head_index])
                mask[valid_indices] = 1
                
                logit = logits[batch_index, :]
                logit[mask == 0] = -float('inf')
            
            dist = Categorical(logits=logits)

            log_prob += dist.log_prob(actions[:, head_index])
            entropy += dist.entropy()

        return values, log_prob, entropy


class MaskedDQNPolicy(DQNPolicy):
    def __init__(self, *args, agent_player=h.Cells.PlayerOne, enable_rotation=True, raise_error=True, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn = 0 if enable_rotation or agent_player == h.Cells.PlayerOne else 1
        self.raise_error = raise_error
        
    def _predict(self, observation, deterministic=True):
        if not self.raise_error:
            return super()._predict(observation, deterministic)

        q_values = self.q_net(observation)
        
        greedy_actions = []
        for batch_number in range(observation.shape[0]):
            single_obs = observation[batch_number]
            as_normal_grid = single_obs.argmax(dim=0)
            valid_move_indices = mg.generate_valid_moves(self.turn, as_normal_grid)
            mask = np.isin(np.arange(single_obs.shape[-1] ** 4 * 6), valid_move_indices)
        
            masked_q = q_values[batch_number]
            masked_q[mask == 0] = -float("inf")
            
            if deterministic:
                greedy_actions.append(masked_q.argmax().item())
            else:
                greedy_actions.append(random.choice(valid_move_indices))
        
        return torch.tensor(greedy_actions, device=q_values.device)

  
def load_model_weights_to_model(
    algorithm_name, env, model_path=None,
    verbose=1, agent_player=h.Cells.PlayerOne, 
    raise_error=False, hex_grid_size=2,
    tensorboard_logs=True, just_tensor=False,
    only_predict=False,
    **model_kwargs
):
    if not only_predict:
        log_path = h.get_logs(algorithm_name, hex_grid_size=hex_grid_size)
        # new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

        match algorithm_name:
            case 'a2c': 
                model = A2C(
                    MaskedPolicyBased, env, verbose=verbose, 
                    policy_kwargs=dict(
                        agent_player=agent_player, raise_error=raise_error, 
                        features_extractor_class=CnnMlpExtractor,
                        features_extractor_kwargs=dict(action_space=env.action_space),
                    ),
                    tensorboard_log=log_path if just_tensor or verbose == 1 and tensorboard_logs else None,
                    device=h.device,
                    **model_kwargs
                )
            case 'a2c_md':
                model = A2C(
                    MDPolicyBasedMasked, env, verbose=verbose, 
                    policy_kwargs=dict(
                        # Might need to put action space here
                        features_extractor_class=CnnMlpExtractor,
                        features_extractor_kwargs=dict(action_space=env.action_space),
                        net_arch=[1024, 1024]
                    ),
                    tensorboard_log=log_path if just_tensor or verbose == 1 and tensorboard_logs else None,
                    device=h.device,
                    **model_kwargs
                )
            case 'ppo':
                model = PPO(
                    MaskedPolicyBased, env, verbose=verbose, 
                    policy_kwargs=dict(
                        agent_player=agent_player, raise_error=raise_error, 
                        features_extractor_class=CnnMlpExtractor, 
                        features_extractor_kwargs=dict(action_space=env.action_space),
                    ),
                    tensorboard_log=log_path if just_tensor or verbose == 1 and tensorboard_logs else None,
                    device=h.device,
                    **model_kwargs
                )
            case 'ppo_md':
                model = PPO(
                    MDPolicyBasedMasked, env, verbose=verbose, 
                    policy_kwargs=dict(
                        features_extractor_class=CnnMlpExtractor, 
                        features_extractor_kwargs=dict(action_space=env.action_space),
                        net_arch=[1024, 1024]
                    ),
                    tensorboard_log=log_path if just_tensor or verbose == 1 and tensorboard_logs else None,
                    device=h.device,
                    **model_kwargs
                )
            case 'dqn':
                model = MaskedDQN(
                    MaskedDQNPolicy, env, verbose=verbose, 
                    policy_kwargs=dict(
                        agent_player=agent_player, 
                        raise_error=raise_error,
                        features_extractor_class=CnnMlpExtractor,
                        features_extractor_kwargs=dict(
                            action_space=env.action_space,
                            is_dqn=True
                        ),
                    ),
                    tensorboard_log=log_path if just_tensor or verbose == 1 and tensorboard_logs else None,
                    device=h.device,
                    **model_kwargs
                )
            case _:
                raise RuntimeError(f"Algorithm name {algorithm_name} is not in list [a2c, a2c_md, ppo, ppo_md, dqn]")
    
    if model_path is not None:
        match algorithm_name:
            case 'a2c': best_model = A2C.load(model_path, device=h.device)
            case 'a2c_md': best_model = A2C.load(model_path, device=h.device)
            case 'ppo': best_model = PPO.load(model_path, device=h.device)
            case 'ppo_md': best_model = PPO.load(model_path, device=h.device)
            case 'dqn': best_model = DQN.load(model_path, device=h.device)
            case _: raise RuntimeError(f"Algorithm name {algorithm_name} is not in list [a2c, a2c_md, ppo, ppo_md, dqn]")
            
        if only_predict:
            return best_model
            
        if algorithm_name != 'dqn':
            model.policy.load_state_dict(best_model.policy.state_dict())
        else:
            model.q_net.load_state_dict(best_model.q_net.state_dict())
            model.q_net_target.load_state_dict(best_model.q_net_target.state_dict())
                
    return model
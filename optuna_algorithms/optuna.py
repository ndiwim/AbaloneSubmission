import helper as h
import numpy as np
import models.gamesHelper as gh
import models.modelSingleton as ms
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy


'''
    We should be getting ranges from paper. However, we have specific use cases for some 
    
    SHARED
        gamma
            This is the discount factor.
            how long term it thinks.
            No use ranging from 0.1, since we know for a fact that we need to think long term.
            instead, going to be from 0.8 to 0.999
            
        gae_lambda
            something about mt full returns
            long horizon, so same will apply
            
        learning_rate
            straight from paper
            
        ent_coef
            from the paper
            
        vf_coef
            from the paper
            
    A2C
        n_steps
            not added to paper
            we have a logic that 20 moves ahead should be more than enough.
            
    PPO
        clip_range, clip_range_vf 
            taken from paper
'''

def a2c_objective(trial: optuna.trial.Trial):
    algorithm_name = 'a2c_md'
    
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-9, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 0.75)
    n_steps = trial.suggest_int("n_steps", 5, 20)
    
    env = gh.get_env(
        agent_player=h.Cells.PlayerOne, no_envs=28, hex_grid_size=3, 
        model=None, raise_error=True,is_dqn= False, 
        gamma=gamma, is_md=True, 
        is_swapper=True
    )
    
    model: A2C = ms.load_model_weights_to_model(
        algorithm_name=algorithm_name, env=env, model_path=None, 
        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
        hex_grid_size=3,  tensorboard_logs=False, just_tensor=True,
        # Params
        learning_rate=learning_rate, gamma=gamma,
        gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef,
        n_steps=n_steps
    )

    print("################################")
    print("Test")
    print(f"learning_rate: {learning_rate}")
    print(f"gamma: {gamma}")
    print(f"gae_lambda: {gae_lambda}")
    print(f"ent_coef: {ent_coef}")
    print(f"vf_coef: {vf_coef}")
    print(f"n_steps: {n_steps}")
    print("################################")
    
    experiment_description = f'{algorithm_name}\nParams:\n'
    experiment_description += f"learning_rate: {learning_rate}\n"
    experiment_description += f"gamma: {gamma}\n"
    experiment_description += f"gae_lambda: {gae_lambda}\n"
    experiment_description += f"ent_coef: {ent_coef}\n"
    experiment_description += f"vf_coef: {vf_coef}\n"
    experiment_description += f"n_steps: {n_steps}\n"
    
    experiment_name = f'{algorithm_name}_optuna_trial'
    model.learn(
        total_timesteps=300000, 
        log_interval=30, 
        tb_log_name=experiment_name
    )
    
    log_path = gh.prepare_logs(
        algorithm_name, experiment_name, experiment_description, 
        hex_grid_size=3
    )

    h.extract_logs(log_path)
    
    episode_rewards, episode_lengths = evaluate_policy(
        model, env, n_eval_episodes=10, 
        deterministic=False, return_episode_rewards=True
    )

    env.close()
    return calculate_objective(episode_rewards, episode_lengths)

def ppo_objective(trial: optuna.trial.Trial):
    algorithm_name = 'ppo_md'
    
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-9, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 0.75)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    clip_range_vf = trial.suggest_float("clip_range_vf", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [8, 32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    env = gh.get_env(
        agent_player=h.Cells.PlayerOne, no_envs=16, hex_grid_size=3, 
        model=None, raise_error=True,is_dqn= False, 
        gamma=gamma, is_md=True,
        is_swapper=True
    )
    
    model: PPO = ms.load_model_weights_to_model(
        algorithm_name=algorithm_name, env=env, model_path=None, 
        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
        hex_grid_size=3, tensorboard_logs=False, n_epochs=3,
        just_tensor=True,
        # Params
        learning_rate=learning_rate, gamma=gamma,
        gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef,
        n_steps=n_steps, batch_size=batch_size,
        clip_range_vf=clip_range_vf, clip_range=clip_range
    )

    print("################################")
    print("Test")
    print(f"learning_rate: {learning_rate}")
    print(f"gamma: {gamma}")
    print(f"gae_lambda: {gae_lambda}")
    print(f"ent_coef: {ent_coef}")
    print(f"vf_coef: {vf_coef}")
    print(f"clip_range: {clip_range}")
    print(f"clip_range_vf: {clip_range_vf}")
    print(f"n_steps: {n_steps}")
    print(f"batch_size: {batch_size}")
    print("################################")
    
    experiment_description = f'{algorithm_name}\nParams:\n'
    experiment_description += f"learning_rate: {learning_rate}\n"
    experiment_description += f"gamma: {gamma}\n"
    experiment_description += f"gae_lambda: {gae_lambda}\n"
    experiment_description += f"ent_coef: {ent_coef}\n"
    experiment_description += f"vf_coef: {vf_coef}\n"
    experiment_description += f"clip_range: {clip_range}\n"
    experiment_description += f"clip_range_vf: {clip_range_vf}\n"
    experiment_description += f"n_steps: {n_steps}\n"
    experiment_description += f"batch_size: {batch_size}\n"
    
    experiment_name = f'{algorithm_name}_optuna_trial'
    model.learn(
        total_timesteps=300000, 
        log_interval=20, 
        tb_log_name=experiment_name
    )
    
    log_path = gh.prepare_logs(
        algorithm_name, experiment_name, experiment_description, 
        hex_grid_size=3
    )

    h.extract_logs(log_path)
    
    episode_rewards, episode_lengths = evaluate_policy(
        model, env, n_eval_episodes=10, 
        deterministic=False, return_episode_rewards=True
    )

    env.close()
    return calculate_objective(episode_rewards, episode_lengths)

def dqn_objective(trial: optuna.trial.Trial):
    algorithm_name = 'dqn'
    
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    train_freq = trial.suggest_categorical("train_freq", [4, 16, 32, 64])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
    n_steps = trial.suggest_categorical("n_steps", [2, 4, 8])
    target_update_interval = trial.suggest_categorical("target_update_interval", [250, 500, 1000, 2000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    env = gh.get_env(
        agent_player=h.Cells.PlayerOne, no_envs=16, hex_grid_size=3, 
        model=None, raise_error=True,is_dqn=True, 
        gamma=gamma, is_md=False,
        is_swapper=True
    )
    
    model: DQN = ms.load_model_weights_to_model(
        algorithm_name=algorithm_name, env=env, model_path=None, 
        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
        just_tensor=True,
        # Params
        hex_grid_size=3, tensorboard_logs=False, train_freq=train_freq,
        learning_rate=learning_rate, gamma=gamma, gradient_steps=gradient_steps,
        n_steps=n_steps, target_update_interval=target_update_interval, batch_size=batch_size,
        exploration_fraction=exploration_fraction
    )

    experiment_description = f'{algorithm_name}\nParams:\n'
    experiment_description += f"learning_rate: {learning_rate}\n"
    experiment_description += f"exploration_fraction: {exploration_fraction}\n"
    experiment_description += f"gamma: {gamma}\n"
    experiment_description += f"train_freq: {train_freq}\n"
    experiment_description += f"gradient_steps: {gradient_steps}\n"
    experiment_description += f"n_steps: {n_steps}\n"
    experiment_description += f"target_update_interval: {target_update_interval}\n"
    experiment_description += f"batch_size: {batch_size}\n"
    
    experiment_name = f'{algorithm_name}_optuna_trial'
    model.learn(
        total_timesteps=300000, 
        log_interval=20, 
        tb_log_name=experiment_name
    )
    
    log_path = gh.prepare_logs(
        algorithm_name, experiment_name, experiment_description, 
        hex_grid_size=3
    )

    h.extract_logs(log_path)

    episode_rewards, episode_lengths = evaluate_policy(
        model, env, n_eval_episodes=10, 
        deterministic=False, return_episode_rewards=True
    )

    env.close()
    return calculate_objective(episode_rewards, episode_lengths)

# Honestly hard coded for 3x3x3, and I think thats fine
def find_best_hyperparameters(algorithm_name, is_retry=False):
    storage = f"sqlite:///{algorithm_name}.db"
    
    match algorithm_name:
        case 'ppo': objective = ppo_objective
        case 'ppo_md': objective = ppo_objective
        case 'a2c': objective = a2c_objective
        case 'a2c_md': objective = a2c_objective
        case 'dqn': objective = dqn_objective
        case _: raise RuntimeError(f"Algorithm name {algorithm_name} is not in list [a2c, a2c_md, ppo, ppo_md, dqn]")
        
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(),
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
    )
    
    if is_retry:
        study_name_from_out = None  
        
        if study_name_from_out is None:
            raise RuntimeError("forgot to manually set the study name")

        # Load old study
        old_study = optuna.load_study(
            study_name_from_out,
            storage=storage
        )
        
        # Find completed trials and add them to the normal study
        for trial in old_study.trials:
            if trial.state == TrialState.COMPLETE:
                study.add_trial(trial)
    
    study.optimize(objective, n_trials=50, n_jobs=1)
    
    
def load_study(algorithm_name):
    return optuna.load_study(
        None,
        storage=f"sqlite:///{algorithm_name}.db"
    )
    
def calculate_objective(mean_rewards, mean_length):
    mean_episode_reward = np.array(mean_rewards).mean()
    mean_episode_length = np.array(mean_length).mean()
    
    if mean_episode_reward > 0:
        return mean_episode_length
    else:
        return 220 - min(mean_episode_length, 120)
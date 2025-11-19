import json
import os
import re
import random
import psutil
import pygame
import helper as h
import numpy as np
import gymenv.moveGenerator as mg
from stable_baselines3 import DQN, PPO, A2C
from models.maskedDQN import MaskedDQN
from gymenv.abaloneEnv import Abalone, draw_box_grid, evaluate_grid
from gymenv.abaloneEnvDiscretePolicyBased import AbaloneEnvDiscretePolicyBased
from gymenv.abaloneEnvMultiDiscrete import AbaloneEnvMultiDiscrete
from gymenv.abaloneEnvDiscrete import AbaloneDiscrete
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import models.modelSingleton as ms
from typing import List
from typing import Dict

'''
    PPO hyperparameter information
        rollout = n_steps * n_envs
            So, if you want longer game information, increase n_steps
            For the best of both worlds, keep n_envs high as well
            
        batch_size
            must divide rollout
            larger is more stable, but less frequent updates
            higher is good for unstable learning

        n_epochs
            how many times it uses the same data to update.
            for times where more recent data is important (during self learning), its better to be high
            few: 2-4
            many: 6-12

        learning rate
            consider looking into decaying schedule
            
        target_kl, clip_range
            look into this combo later
            
        gamma
            how much future rewards count
            for abalone, might be worth making this larger
            0.995
            
        gae_lambda
            Sparse rewards / long horizons: set λ high (0.95-0.99, even 0.995) to better propagate terminal signals across steps via GAE
        
        ent_coef
            controls how much it explores
            watched a video
            maybe we can explore between 0.01, and 0.0001
            
        
        
'''

class SaveModelCallback(BaseCallback):
    def __init__(
        self, save_frequency, total_time_steps, algorithm_name, agent_player, 
        hex_grid_size=2, starting_time_step=0, verbose = 1, 
        p_start=0.8, p_end=0.2, p_duration=1,
        learn_log: h.LearnLog=None,
    ):
        super(SaveModelCallback, self).__init__(verbose)
        self.hex_grid_size = hex_grid_size
        
        self.learn_log = learn_log
        if self.learn_log.enabled:
            self.learn_log.interval_step = total_time_steps // self.learn_log.interval
        
        self.save_frequency = save_frequency
        self.algorithm_name = algorithm_name
        self.starting_time_step = starting_time_step
        self.total_time_steps = total_time_steps
        
        self.current_p = p_start
        self.p_start=p_start
        self.p_end=p_end
        self.p_range = p_start - p_end
        self.p_duration=p_duration
        
        self.step_counter = 0
        self.prev_time_step = 0    
        self.step_counter_game_runner = 0
        self.model_path = h.get_train_path(algorithm_name, hex_grid_size=hex_grid_size)
        
        self.loaded_models = {}
        self.load_models()
    
    def load_models(self):
        model_list = h.get_model_list(self.algorithm_name, hex_grid_size=self.hex_grid_size)
        for model_path in model_list:
            model_name_pattern = r"[^\/\\]+\.zip"
            regex_match = re.search(model_name_pattern, model_path)
            if regex_match:
                model_name = regex_match.group(0)
                
                self.loaded_models[model_name] = ms.load_model_weights_to_model(
                    algorithm_name=self.algorithm_name, env=None, 
                    model_path=model_path, verbose=0, agent_player=h.Cells.PlayerOne, 
                    raise_error=True, hex_grid_size=self.hex_grid_size, 
                    tensorboard_logs=False, just_tensor=False, only_predict=True
                )
    
    def _init_callback(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.step_counter = 0
        self.prev_time_step = 0
        self.current_p = self.p_start
        
    def _on_step(self):
        no_steps_past = self.num_timesteps - self.prev_time_step
        
        self.step_counter += no_steps_past
        self.step_counter_game_runner += no_steps_past
        
        current_progress = self.num_timesteps / (self.total_time_steps * self.p_duration)
        self.current_p = self.p_start - self.p_range * min(current_progress, 1)
        
        if self.save_frequency != 0 and self.step_counter >= self.save_frequency:
            step_checkpoint = divmod(self.num_timesteps, self.save_frequency)[0]
            model_save_timestep = f"{self.algorithm_name}_{step_checkpoint * self.save_frequency + self.starting_time_step}"
            self.model.save(os.path.join(self.model_path, model_save_timestep))
            self.step_counter -= self.save_frequency
            
            model_save_timestep += '.zip'
            self.loaded_models[model_save_timestep] = ms.load_model_weights_to_model(
                algorithm_name=self.algorithm_name, env=None, 
                model_path=os.path.join(self.model_path, model_save_timestep), verbose=0, agent_player=h.Cells.PlayerOne, 
                raise_error=True, hex_grid_size=self.hex_grid_size, 
                tensorboard_logs=False, just_tensor=False, only_predict=True
            )
        
        if self.learn_log.enabled:
            if self.step_counter_game_runner >= self.learn_log.interval_step:
                
                self.step_counter_game_runner -= self.learn_log.interval_step
                win_record = simulate_games(self.algorithm_name, self.model, self.learn_log)
                
                step_checkpoint = divmod(self.num_timesteps, self.learn_log.interval_step)[0]
                self.learn_log.history[f'{step_checkpoint * self.learn_log.interval_step}'] = list.copy(win_record)
            
                
        self.prev_time_step = self.num_timesteps
        return True


h.set_seeds()

 
def get_env(
    agent_player, no_envs=1, hex_grid_size=2, 
    model: h.OpponentClass|None=None, raise_error=True, is_dqn=False, gamma=0.99,
    is_md=False, is_swapper=False
):
    if no_envs < 2:
        if is_dqn:
            return AbaloneDiscrete(
                Abalone(
                    raise_error=raise_error, 
                    hex_grid_size=hex_grid_size, 
                    agent_player=agent_player, 
                    is_agent=True, 
                    model=model,
                ),
                agent_player,
                gamma=gamma,
                enable_swapping=is_swapper,
            )
        elif is_md:
            return AbaloneEnvMultiDiscrete(
                Abalone(
                    raise_error=False, 
                    hex_grid_size=hex_grid_size, 
                    agent_player=agent_player, 
                    is_agent=True, 
                    model=model,
                ),
                agent_player,
                gamma=gamma,
                enable_swapping=is_swapper,
            )
        else:
            return AbaloneEnvDiscretePolicyBased(
                Abalone(
                    raise_error=raise_error, 
                    hex_grid_size=hex_grid_size, 
                    agent_player=agent_player, 
                    is_agent=True, 
                    model=model,
                ),
                agent_player,
                gamma=gamma,
                enable_swapping=is_swapper,
            )
    else:
        if is_dqn:
            return make_vec_env(
                lambda: AbaloneDiscrete(
                    Abalone(
                        raise_error=raise_error, 
                        hex_grid_size=hex_grid_size, 
                        agent_player=agent_player, 
                        is_agent=True,
                        model=model,
                    ),
                    agent_player,
                    gamma=gamma,
                    enable_swapping=is_swapper,
                ), 
                n_envs=no_envs,
                
            )
        elif is_md:
            return make_vec_env(
                lambda: AbaloneEnvMultiDiscrete(
                    Abalone(
                        raise_error=False, 
                        hex_grid_size=hex_grid_size, 
                        agent_player=agent_player, 
                        is_agent=True,
                        model=model,
                    ),
                    agent_player,
                    gamma=gamma,
                    enable_swapping=is_swapper,
                ), 
                n_envs=no_envs,
            )
        else:
            return make_vec_env(
                lambda: AbaloneEnvDiscretePolicyBased(
                    Abalone(
                        raise_error=raise_error, 
                        hex_grid_size=hex_grid_size, 
                        agent_player=agent_player, 
                        is_agent=True,
                        model=model,
                    ),
                    agent_player,
                    gamma=gamma,
                    enable_swapping=is_swapper,
                ), 
                n_envs=no_envs,
            )
          
 
def learn(
    algorithm_name, experiment_name, experiment_description, 
    agent_player=h.Cells.PlayerOne, no_envs=1, 
    hex_grid_size=2, raise_error=True, 
    total_time_steps=100000, continue_model=False, 
    self_play=False, run_games=False, run_game_interval=0,
    run_game_rounds=1, model_save_frequency=10,
    **model_kwargs
):
    is_swapper = agent_player == h.Cells.Empty
    agent_player = h.Cells.PlayerOne if is_swapper else agent_player
    
    if re.search(r"\s", algorithm_name):
        raise RuntimeError(f"Experiment name: {experiment_name}, should not have a space")

    env = get_env(
        agent_player=agent_player, no_envs=no_envs, 
        hex_grid_size=hex_grid_size, raise_error=raise_error,
        is_swapper=is_swapper, is_dqn='dqn' in algorithm_name, 
        is_md="_md" in algorithm_name
    )
    
    starting_time_step = 0
    best_model_str = None
    
    if continue_model:
        best_model_str, best_model_time_step = h.get_best_model(algorithm_name, hex_grid_size=hex_grid_size)
        if best_model_str is not None:
            starting_time_step = best_model_time_step
            
    model = ms.load_model_weights_to_model(
        algorithm_name, env, model_path=best_model_str, 
        verbose=1, agent_player=agent_player, raise_error=raise_error, 
        hex_grid_size=hex_grid_size, **model_kwargs
    )        
    
    learn_log = h.LearnLog(enabled=run_games, interval=run_game_interval, no_rounds=run_game_rounds) 
    
    save_frequency = 0 if model_save_frequency == 0 else total_time_steps // model_save_frequency
    callback = SaveModelCallback(
        save_frequency, total_time_steps, algorithm_name, 
        agent_player, hex_grid_size=hex_grid_size, 
        starting_time_step=starting_time_step,
        learn_log=learn_log, 
    )
    
    if self_play:
        if hasattr(env, "envs"):
            for env_container in env.envs:
                env_container.env.env.set_callback(callback)
        else:
            env.env.set_callback(callback)
    
    match algorithm_name:
        case 'a2c': log_interval = 30
        case 'a2c_md': log_interval = 30
        case 'ppo_md': log_interval = 1
        case 'ppo': log_interval = 1
        case 'dqn': log_interval = 10
        case _: raise RuntimeError(f"Algorithm name {algorithm_name} is not in list [a2c, a2c_md, ppo, ppo_md, dqn]")
        
    model.learn(
        total_timesteps=total_time_steps, 
        log_interval=log_interval, 
        callback=callback,
        tb_log_name=experiment_name
    )
    
    log_path = prepare_logs(
        algorithm_name, experiment_name, experiment_description, 
        hex_grid_size=hex_grid_size, learn_log=learn_log
    )

    h.extract_logs(log_path)


def play(
    algorithm_name, model_path=None, agent_player=h.Cells.PlayerOne, 
    hex_grid_size=2, seed=42, raise_error=True, agent: h.OpponentClass=None, 
    opponent: h.OpponentClass=None, deterministic=True
):
    is_dqn = 'dqn' in algorithm_name
    is_md="_md" in algorithm_name
    
    env = get_env(
        agent_player=agent_player, hex_grid_size=hex_grid_size,
        is_dqn=is_dqn, is_md=is_md, model=opponent
    )

    best_model_str = h.get_best_model(algorithm_name, hex_grid_size=hex_grid_size)[0]
    if model_path is None and best_model_str is None:
        train_path = h.get_train_path(algorithm_name, hex_grid_size=hex_grid_size)
        raise RuntimeError(f"Tried to load best model for play, but no models were found in {train_path}") 
    else:
        selected_model_path = model_path if model_path is not None else best_model_str
        model = ms.load_model_weights_to_model(
            algorithm_name, env, model_path=selected_model_path, 
            verbose=0, agent_player=agent_player, raise_error=raise_error, 
            hex_grid_size=hex_grid_size
        )
    
    print("Loading Game")
    h.set_seeds(seed=seed)
    env.reset(seed=seed)
    while True:
        if env.env.current_player_to_move() == agent_player:    
            if agent_player == h.Cells.PlayerTwo:
                action = model.predict(env.env.encoded_rotated_state, deterministic=deterministic)[0]
            else:
                action = model.predict(env.env.encoded_state, deterministic=deterministic)[0]
            _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        
    move_buffer = env.env.move_buffer
    state_buffer = []
    
    env = Abalone(
        render_mode=h.RenderModes.BoxPygame, hex_grid_size=hex_grid_size, 
        raise_error=True, is_agent=False, agent_player=h.Cells.PlayerOne, 
        model=None
    )
    
    # Gathering state transitions
    print("Done loading game.\nNow setting up move buffer")
    state_buffer.append(env.reset()[0].copy())

    for move in move_buffer:
        obs, _, _, _, _ = env.step(move)
        state_buffer.append(obs.copy())

    print("Starting graphical game")
    # Now we can draw the board and all
    move_index = 0
    prev_move_index = -1
    prev_eval = 0
    run_game = True
    while run_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run_game = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    move_index = max(0, move_index - 1)
                elif event.key == pygame.K_RIGHT:
                    move_index = min(len(move_buffer), move_index + 1)
        
        draw_box_grid(env.screen, state_buffer[move_index], env.font)
        if prev_move_index != move_index:
            prev_move_index = move_index

            eval = evaluate_grid(state_buffer[move_index], h.Cells.PlayerOne)
            delta = eval - prev_eval
            prev_eval = move_index
            prev_eval = eval
            print(f"Eval: {round(eval, 5)}\nDelta: {round(delta, 5)}\n")

    env.close()
     

def prepare_logs(algorithm_name, experiment_name, experiment_description, hex_grid_size, learn_log: h.LearnLog | None=None):
    if re.search(r"\s", algorithm_name):
        raise RuntimeError(f"Experiment name: {experiment_name}, should not have a space")

    base_log_path = h.get_logs(algorithm_name, hex_grid_size)
    log_path = os.path.join(base_log_path, f"{experiment_name}_1")

    '''
        If the log_path exists, use the _1, 2, 3 system
    '''
    if os.path.isdir(log_path):
        max_int = None
        for directory in os.listdir(base_log_path):
            pattern = rf'({experiment_name})_(\d+)'
            match = re.match(pattern, directory)
            if match:
                occurrence_no = int(match.group(2))
                max_int = occurrence_no if max_int is None else max(max_int, occurrence_no)

        if max_int is not None:
            log_path = os.path.join(base_log_path, f"{experiment_name}_{max_int}")
    
    with open(os.path.join(log_path, "description.txt"), 'w') as file:
        file.write(experiment_description)
        
        if learn_log is not None and learn_log.enabled:
            file.write(json.dumps(learn_log.history))
    
    return log_path


def simulate_games(algorithm_name, current_model, learn_log: h.LearnLog):
    env = Abalone(
        render_mode=None, hex_grid_size=3, raise_error=True, 
        is_agent=False, agent_player=h.Cells.PlayerOne, 
        model=None
    )
    
    if algorithm_name == 'dqn':    
        agent_env = AbaloneDiscrete(
            env,
            h.Cells.PlayerOne,
            gamma=0.99,
            enable_swapping=False,
        )
        
        agent = ms.load_model_weights_to_model(
            algorithm_name='dqn', env=agent_env, model_path=None, 
            verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
            hex_grid_size=3, tensorboard_logs=False
        )
    else:
        agent_env = AbaloneEnvMultiDiscrete(
            env,
            h.Cells.PlayerOne,
            gamma=0.99,
            enable_swapping=False,
        )

        if 'a2c' in algorithm_name:
            agent = ms.load_model_weights_to_model(
                algorithm_name=algorithm_name, env=agent_env, model_path= None,
                verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True,
                hex_grid_size=3, tensorboard_logs=False
            )
        else:
            agent = ms.load_model_weights_to_model(
                algorithm_name=algorithm_name, env=agent_env, model_path=None,
                verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True,
                hex_grid_size=3, tensorboard_logs=False
            )
    
    
    if algorithm_name != 'dqn':
        agent.policy.load_state_dict(current_model.policy.state_dict())
    else:
        agent.q_net.load_state_dict(current_model.q_net.state_dict())
        agent.q_net_target.load_state_dict(current_model.q_net_target.state_dict())
        
    # White wins / Black wins
    win_record = [0, 0]        
    for color in [h.Cells.PlayerOne, h.Cells.PlayerTwo]:
        for _ in range(learn_log.no_rounds):
            env.reset()
            winner = None
            
            while True:
                is_random = False
                moving_player = env.current_player_to_move()
                
                if moving_player == color:
                    if color == h.Cells.PlayerOne:
                        action = agent.predict(observation=env.encoded_state, deterministic=False)[0]
                    else:
                        action = agent.predict(observation=env.encoded_rotated_state, deterministic=False)[0]
                else:
                    action = random.choice(mg.generate_valid_moves(env.turn, state=env.state))
                    is_random = True
                    
                if not is_random and 'md' in algorithm_name:
                    action = h.multi_discrete_to_action(action, grid_size=5)
                else:
                    action = h.index_to_action(action, grid_size=5)
                    
                _, _, terminated, truncated, _ = env.step(action, act_on_rotated_grid=not is_random and moving_player == h.Cells.PlayerTwo)
                
                if terminated or truncated:
                    winner = env.winner
                    break
            
            if winner == color:
                win_record[color - 1] += 1
    
    return win_record
    

def round_robin(no_rounds):
    best_dqn_model_path = h.get_best_model(algorithm_name="dqn", hex_grid_size=3, train_path=None)[0]
    best_ppo_md_model_path = h.get_best_model(algorithm_name='ppo_md', hex_grid_size=3, train_path=None)[0]
    best_a2c_md_model_path = h.get_best_model(algorithm_name='a2c_md', hex_grid_size=3, train_path=None)[0]
    
    print(f"DQN Model: {best_dqn_model_path}")
    print(f"A2C Model: {best_a2c_md_model_path}")
    print(f"PPO Model: {best_ppo_md_model_path}")
    
    env = Abalone(
        render_mode=None, hex_grid_size=3, raise_error=True, 
        is_agent=False, agent_player=h.Cells.PlayerOne, 
        model=None
    )

    
    dqn_env = AbaloneDiscrete(
        env,
        h.Cells.PlayerOne,
        gamma=0.99,
        enable_swapping=False,
    )
    
    policy_env = AbaloneEnvMultiDiscrete(
        env,
        h.Cells.PlayerOne,
        gamma=0.99,
        enable_swapping=False,
    )
    
    dqn_agent = ms.load_model_weights_to_model(
        algorithm_name='dqn', env=dqn_env, model_path=best_dqn_model_path, 
        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
        hex_grid_size=3, tensorboard_logs=False
    )
    
    ppo_md_agent = ms.load_model_weights_to_model(
        algorithm_name='ppo_md', env=policy_env, model_path=best_ppo_md_model_path,
        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True,
        hex_grid_size=3, tensorboard_logs=False
    )
    a2c_md_agent = ms.load_model_weights_to_model(
        algorithm_name='a2c_md', env=policy_env, model_path= best_a2c_md_model_path,
        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True,
        hex_grid_size=3, tensorboard_logs=False
    )
    
    win_index = 0
    loss_index = 1
    
    agents = [None, ppo_md_agent, a2c_md_agent, dqn_agent]
    action_types = ["discrete", "md", "md", "discrete"]
    agent_names = ['random', 'ppo', 'a2c', 'dqn']
    
    empty_result = np.zeros(2, dtype=np.int64)
    results = np.zeros((4, 4, 2), dtype=np.int64)
    
    for active_model in range(4):
        for opponent_model in range(4):
            print(f"{agent_names[active_model]} vs {agent_names[opponent_model]}")
            result_entry = results[active_model, opponent_model, :].squeeze()
            if active_model != opponent_model and np.equal(result_entry , empty_result).all():
                for color in [h.Cells.PlayerOne, h.Cells.PlayerTwo]:
                    for round in range(no_rounds):
                        env.reset()
                        winner = None
                        
                        while True:
                            moving_player = env.current_player_to_move()
                            controlling_agent_index = active_model if moving_player == color else opponent_model
                            
                            is_random = agents[controlling_agent_index] is None
                                
                            if is_random:
                                action = random.choice(mg.generate_valid_moves(env.turn, state=env.state))
                            else:
                                if moving_player == h.Cells.PlayerOne:
                                    action = agents[controlling_agent_index].predict(observation=env.encoded_state, deterministic=False)[0]
                                else:
                                    action = agents[controlling_agent_index].predict(observation=env.encoded_rotated_state, deterministic=False)[0]
                                
                            match (action_types[controlling_agent_index]):
                                case "md": action = h.multi_discrete_to_action(action, grid_size=5)
                                case "discrete": action = h.index_to_action(action, grid_size=5)
                                case _: raise RuntimeError("Nahhh")
                                
                            _, _, terminated, truncated, _ = env.step(action, act_on_rotated_grid=not is_random and moving_player == h.Cells.PlayerTwo)
                            
                            if terminated or truncated:
                                winner = env.winner
                                break
                        
                        if winner == color:
                            results[active_model, opponent_model, win_index] += 1
                            results[opponent_model, active_model, loss_index] += 1
                        elif winner is None:
                            raise RuntimeError('winner result is weird...')
                        else:
                            results[opponent_model, active_model, win_index] += 1
                            results[active_model, opponent_model, loss_index] += 1
                        
    for row in range(4):
        line = ''
        for col in range(4):
            line += f'{results[row, col]}'
        print(line)
                            
        
def relearn_models(alg_name=None):
    class OptunaTrials:
        def __init__(self, trial_number, value, params):
            self.trial_number = int(trial_number)
            self.value = float(value)
            self.params = {}
            
            matches = re.findall(r"'([^']+)':\s*([\d.+-eE]+)", params)
            
            for key, param_value in matches:
                self.params[f'{key}'] = float(param_value.strip().rstrip(','))
    
    allowed_trials = [0, 2, 4, 9, 14, 19, 29, 39, 44, 49]
    algorithm_names = ['a2c_md', 'dqn', 'ppo_md']
    file_names = ['A2C_OPT.err', 'DQN_OPT.err', 'PPO_OPT.err']
    trials: Dict[str, List[OptunaTrials]] = {}
    
    pattern_trial_line = r"^.*\btrial\b.*$"
    pattern_trial_details = r"Trial (\d+) finished with value: ([\d.+-eE]+) and parameters: \{(.+?)\}"
    
    for algorithm_name, file_name in zip(algorithm_names, file_names):
        if alg_name is not None and algorithm_name != alg_name:
            continue
            
        with open(os.path.join("err", file_name)) as file:
            contents = file.read()

        matches = re.findall(pattern_trial_line, contents, re.MULTILINE)
        if len(matches) == 0:
            raise RuntimeError(f"Failed to find any trials for {algorithm_name}")

        current_trials: List[OptunaTrials] = []
        for line in matches:
            match = re.search(pattern_trial_details, line)
            
            if match:
                current_trials.append(OptunaTrials(match.group(1), match.group(2), match.group(3)))
            else:
                print(line)
                raise RuntimeError(f"For model {algorithm_name}, failed to find trial values")

        trials[f'{algorithm_name}'] = sorted(current_trials, key=lambda trial: (trial.value, -trial.trial_number))
        
        print(f"{algorithm_name}")
        for trial in trials[f'{algorithm_name}']:
            print(f"{trial.trial_number} -> {trial.value}")
        print("")
            
    exit(99)
        
    for algorithm_name in algorithm_names:
        if alg_name is not None and algorithm_name != alg_name:
            continue
        
        is_dqn = 'dqn' in algorithm_name
        is_md = 'md' in algorithm_name
        
        print(f"busy with algorithm: {algorithm_name}")
        
        counter = 0
        for trial in trials[f'{algorithm_name}']:
            print(f"Number {counter} in list")
            counter += 1
            
            if counter - 1 not in allowed_trials:
                continue
            
            no_envs = 28 if algorithm_name == 'a2c_md' else 16
            
            env = get_env(
                agent_player=h.Cells.PlayerOne, no_envs=no_envs, hex_grid_size=3, 
                model=None, raise_error=True,is_dqn=is_dqn, 
                gamma=trial.params['gamma'], is_md=is_md,
                is_swapper=True
            )
            
            match algorithm_name:
                case 'dqn': 
                    log_interval = 20
                    
                    model: DQN = ms.load_model_weights_to_model(
                    algorithm_name=algorithm_name, env=env, model_path=None, 
                    verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
                    hex_grid_size=3, tensorboard_logs=False, just_tensor=True,
                    # Params
                    train_freq=int(trial.params['train_freq']),
                    learning_rate=trial.params['learning_rate'], 
                    gamma=trial.params['gamma'], 
                    gradient_steps=int(trial.params['gradient_steps']),
                    n_steps=int(trial.params['n_steps']), 
                    target_update_interval=int(trial.params['target_update_interval']), 
                    batch_size=int(trial.params['batch_size']),
                    exploration_fraction=trial.params['exploration_fraction']
                )
                case 'ppo_md':
                    log_interval = 20
                    
                    model: PPO = ms.load_model_weights_to_model(
                        algorithm_name=algorithm_name, env=env, model_path=None, 
                        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
                        hex_grid_size=3, tensorboard_logs=False, n_epochs=3, just_tensor=True,
                        # Params
                        learning_rate=trial.params['learning_rate'], 
                        gamma=trial.params['gamma'],
                        gae_lambda=trial.params['gae_lambda'], 
                        ent_coef=trial.params['ent_coef'], 
                        vf_coef=trial.params['vf_coef'],
                        n_steps=int(trial.params['n_steps']), 
                        batch_size=int(trial.params['batch_size']),
                        clip_range_vf=trial.params['clip_range_vf'], 
                        clip_range=trial.params['clip_range']
                    )
                case 'a2c_md': 
                    log_interval = 30
                    
                    model: A2C = ms.load_model_weights_to_model(
                        algorithm_name=algorithm_name, env=env, model_path=None, 
                        verbose=0, agent_player=h.Cells.PlayerOne, raise_error=True, 
                        hex_grid_size=3,  tensorboard_logs=False, just_tensor=True,
                        # Params
                        learning_rate=trial.params['learning_rate'], 
                        gamma=trial.params['gamma'],
                        gae_lambda=trial.params['gae_lambda'], 
                        ent_coef=trial.params['ent_coef'], 
                        vf_coef=trial.params['vf_coef'],
                        n_steps=int(trial.params['n_steps'])
                    )
                case _: raise RuntimeError("Algorithm name not in list [a2c_md, ppo_md, dqn]")    
            
            
            experiment_description = f'{algorithm_name}\nTrial: {trial.trial_number}\nNumber: {counter}\nParams:\n'
            for key in trial.params.keys():
                experiment_description += f'{key}: {trial.params[f'{key}']}\n'
            
            experiment_name = f'{algorithm_name}_trial_{counter}_{trial.trial_number}'
            model.learn(
                total_timesteps=300000, 
                log_interval=log_interval, 
                tb_log_name=experiment_name
            )
            
            log_path = prepare_logs(
                algorithm_name, experiment_name, experiment_description, 
                hex_grid_size=3
            )

            h.extract_logs(log_path)
import sys
import helper as h
import numpy as np
from models.gamesHelper import learn, play, round_robin, relearn_models
import gymenv.moveGenerator as mg
import optuna_algorithms.optuna as op
import optuna
from gymenv.abaloneEnv import evaluate_grid

'''
    TODO
    Could look into adding gamma into learn in game helper
    
    we can speed up the rotational moves by performing the inverse of the current action on the rotated grid
    
    perform some experiments to see if you can rather do that multi dimensional action
        that way the agent can maybe learn something more useful

    Look into, when playing against the model in the env, to control if it should be deterministic or not

    Currently running experiments to see if we can make the A2C more clinical, it's not scared of losing pieces clearly.
        Its not hungry to end the game as soon as possible
        
    set_parameters could be used for loading those for the models instead
    
    1024 is hard coded in the md stuff    
'''

if __name__ == "__main__":
    print(f"Device: {h.device}")

    #################################################################
    '''
        Learn Code
    '''
    #################################################################

    description = '''
        This is a description that will be there after learning
    '''

    learn(
        'a2c_md', 'code_demo', description,
        h.Cells.Empty, no_envs=1, hex_grid_size=3, 
        raise_error=True, total_time_steps=300000,
        continue_model=False, self_play=False,
        model_save_frequency=0,
        
    #     run_games=True, run_game_interval=1000, run_game_rounds=25,
        
        # PPO Model Kwargs
        # learning_rate = 4.976990056643746e-05,
        # gamma = 0.8470729823388848,
        # gae_lambda = 0.9004963990475493,
        # ent_coef = 4.1542763828366756e-09,
        # vf_coef = 0.6086185858028199,
        # clip_range = 0.3295536273487319,
        # clip_range_vf = 0.14644768288651644,
        # n_steps = 64,
        # batch_size = 64,
        # n_epochs=3
        
        # A2C Model Kwargs
        learning_rate = 0.000433379810447878,
        gamma = 0.8326490013242785,
        gae_lambda = 0.8801603467185235,
        ent_coef = 0.00010768571979758323,
        vf_coef = 0.2505557112631042,
        n_steps = 7
        
        # DQN Model Kwargs
    #     learning_rate = 0.00021356607650638312,
    #     exploration_fraction = 0.356061825834261,
    #     gamma = 0.8646787057926524,
    #     train_freq = 4,
    #     gradient_steps = 2,
    #     n_steps = 4,
    #     target_update_interval = 500,
    #     batch_size = 128
    )


    #################################################################
    '''
        Play Code
    '''
    ################################################################
    # opp_algorithm_name = 'ppo_md'
    # opponent = h.OpponentClass(
    #     model_path=h.get_best_model(opp_algorithm_name, 3)[0],
    #     algorithm_name=opp_algorithm_name,
    #     raise_error=True
    # )
    # play(
    #     'ppo_md', model_path=None, agent_player=h.Cells.PlayerOne, 
    #     hex_grid_size=3, seed=21, raise_error=True, 
    #     agent=None, opponent=opponent, deterministic=False
    # )
    
    #################################################################
    '''
        Optuna Code
    '''
    #################################################################
    
    # Learn
    # op.find_best_hyperparameters('dqn')
    
    # Eval
    # study: optuna.study.Study = op.load_study('a2c_md')
    # for trial in study.get_trials():
    #     print("Trial number:", trial.number)
    #     print("Objectives:", trial.values)  # list of objectives, in the order you defined
    #     print("Hyperparameters:", trial.params)
    #     print("---")
    
    # Checking cache effectiveness
    # print(mg.generate_valid_moves_cached.cache_info())
    
    # relearn_models()
    
    #################################################################
    '''
        Tournament Code
    '''
    #################################################################
    # round_robin(500)
    
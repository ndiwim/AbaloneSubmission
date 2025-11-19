import math
import pygame
import random
import numpy as np
import helper as h
import gymnasium as gym
from typing import Tuple
from gymnasium import spaces
import gymenv.moveGenerator as mg
from stable_baselines3 import DQN, PPO, A2C
import models.modelSingleton as ms

# Optimize step, no need to copy state all the time
# Cache direction vectors

class Abalone(gym.Env):
    # metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self, render_mode=None, hex_grid_size=3, raise_error=True,
        is_agent=False, agent_player=h.Cells.PlayerOne, model: h.OpponentClass|None=None
    ):
        super().__init__()
        
        self.callback = None
        self.parent = None
        self.opponent: h.OpponentClass = model
        self.raise_error = raise_error
        
        self.move_buffer = []
        self.agent_player = agent_player
        self.enemy_player = h.Cells.PlayerOne if agent_player == h.Cells.PlayerTwo else h.Cells.PlayerTwo
        self.is_agent = is_agent
        self.winner = -1
        self.radius = hex_grid_size-1
        self.piece_information = {
            h.Cells.PlayerOne: [0, 0, 0],
            h.Cells.PlayerTwo: [0, 0, 0]
        }
        hex_grid = h.make_hex_grid(hex_grid_size, piece_information=self.piece_information)
        self.grid_size = hex_grid.shape[-1]
        
        self.action_space = spaces.Dict({
            "start": spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            "end": spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            "direction": spaces.Discrete(6)
        })
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=np.int32
        )
        
        # Initialize state
        self.state = hex_grid
        self.rotated_state = h.rotate_grid(hex_grid.copy())
        self.encoded_state = h.one_hot_encode_grid(self.state, flatten=False)
        self.encoded_rotated_state = h.one_hot_encode_grid(self.rotated_state, flatten=False)
        self.turn = 0
        self.reward = 0
        self.invalid_move_reasons = []
        
        # Rendering setup
        self.render_mode = h.RenderModes.Grid if render_mode is None else render_mode 
        
        # Pygame setup
        if (self.render_mode in [h.RenderModes.BoxPygame]):    
            pygame.init()
            self.screen = pygame.display.set_mode((700, 700))
            pygame.display.set_caption(f"Abalone {hex_grid_size}x{hex_grid_size}x{hex_grid_size}")
            self.font = pygame.font.SysFont('Arial', 24)
            
            # Todo, add animations if you want
            # self.clock = pygame.time.Clock()
    
    def set_callback(self, callback):
        self.callback = callback
    
    def get_info(self):
        return {
            "turn": self.turn,
            "invalid_move_reasons": self.invalid_move_reasons,
            "piece information": self.piece_information,
            "winner": self.winner
        }

    def reset(self, seed=42, options=None, render_opp_move=False, print_info=False):
        # Initialize random number generator
        super().reset(seed=seed)
        self.action_space.seed(seed)
        
        # Initialize state
        self.move_buffer = []
        self.state = h.make_hex_grid(self.radius + 1, piece_information=self.piece_information)
        self.rotated_state = h.rotate_grid(self.state.copy())
        self.encoded_state = h.one_hot_encode_grid(self.state, flatten=False)
        self.encoded_rotated_state = h.one_hot_encode_grid(self.rotated_state, flatten=False)
        self.turn = 0
        self.winner = -1
        self.reward = 0
        self.invalid_move_reasons = []

        # Setting up opponent model for self play or playing against a bot
        if self.callback is not None:
            model_list = list(self.callback.loaded_models.keys())
            if len(model_list) != 0:
                best_model = model_list.pop()
                best_model_probability = self.callback.current_p
                
                self.opponent = h.OpponentClass(None, self.callback.algorithm_name, self.raise_error)
                if len(model_list) == 0:
                    self.opponent.model = self.callback.loaded_models[best_model]
                else:
                    if random.random() < best_model_probability:
                        self.opponent.model = self.callback.loaded_models[best_model]
                    else:
                        self.opponent.model = self.callback.loaded_models[random.choice(model_list)]
                
        elif self.opponent is not None:
            model_list = h.get_model_list(self.opponent.algorithm_name, hex_grid_size=self.radius + 1)
            if len(model_list) != 0:
                model_path = model_list.pop()
                
                self.opponent.model = ms.load_model_weights_to_model(
                    algorithm_name=self.opponent.algorithm_name, env=self.parent, 
                    model_path=model_path, verbose=0, agent_player=h.get_opp_player(self.agent_player), 
                    raise_error=self.raise_error, hex_grid_size=self.radius + 1, 
                    tensorboard_logs=False, just_tensor=False, only_predict=True
                )
        
        if (self.is_agent and self.current_player_to_move() != self.agent_player):
            self.perform_opponent_move(render_opp_move=render_opp_move, print_info=print_info)
        
        # Return initial observation and info dict
        return self.state, self.get_info()
    
    def render(self, print_info=False):
        
        if self.render_mode == h.RenderModes.Grid:
            h.print_hex_grid(self.state)
            print("")
        elif self.render_mode == h.RenderModes.Human:
            # Implement some pygame drawing function
            pass
        elif self.render_mode == h.RenderModes.BoxPygame:
            draw_box_grid(self.screen, self.state, self.font)
        
        if print_info:
            print(self.get_info())
            print("")
        
    def close(self):
        # Clean up any resources
        if (self.render_mode in [h.RenderModes.BoxPygame]):
            pygame.quit()
        
    def compute_reward(self, prev_state) -> float:
        """
        Computes reward for `current_player` (1 or 2) based on marble losses.
        
        Args:
            current_player: The player whose turn it is (1 or 2).
            next_state: The board state after the action.
            
        Returns:
            float: Reward for the current player.
        """
        
        # Count marbles before and after the action
        prev_p1 = np.sum(prev_state == h.Cells.PlayerOne)
        next_p1 = np.sum(self.state == h.Cells.PlayerOne)
        p1_loss = prev_p1 - next_p1  # Player 1's marbles lost
        
        prev_p2 = np.sum(prev_state == h.Cells.PlayerTwo)
        next_p2 = np.sum(self.state == h.Cells.PlayerTwo)
        p2_loss = prev_p2 - next_p2  # Player 2's marbles lost
        
        # Assign rewards from the perspective of `agent_player`
        if self.agent_player == h.Cells.PlayerOne: 
            return p2_loss - p1_loss
        else:
            return p1_loss - p2_loss
    
    '''
        Not really used hey...
    '''
    def generate_valid_moves(self):
        # h.print_hex_grid(self.state)
        # print("")
        return h.turn_indices_to_actions(
            mg.generate_valid_moves(
                self.turn, self.state
            ), 
            self.grid_size
        )
        
    def perform_action_on_state(self, grid, action, target_cell, enemy_cell, encoded_state=None, update_info=True):
        invalid_move_reasons = self.invalid_move_reasons if update_info else None
        piece_information = self.piece_information if update_info else None
        
        if mg.invalid_start_end(action, grid, invalid_move_reasons):
            return self.runtimeError(action)
        
        direction, magnitude = h.determine_direction(action["start"], action["end"], self.radius)
        
        if mg.check_if_is_invalid_direction(action, direction, magnitude, self.radius, invalid_move_reasons=invalid_move_reasons):
            return self.runtimeError(action)
        
        if not mg.check_selected_cells_are_yours(action, direction, target_cell, grid, self.turn, invalid_move_reasons=invalid_move_reasons):
            return self.runtimeError(action)
        
        is_single_block_move, valid_move = mg.handle_single_block_movement(
            action, self.radius, grid, magnitude=magnitude,
            invalid_move_reasons=invalid_move_reasons,
            encoded_state=encoded_state
        )
        if is_single_block_move and not valid_move:
            return self.runtimeError(action)
        
        if not is_single_block_move:
            if mg.is_push_move(direction, action):
                if not mg.handle_push_logic(
                    action, target_cell, enemy_cell, grid, self.radius, magnitude=magnitude, direction=direction,
                    invalid_move_reasons=invalid_move_reasons, piece_information=piece_information, 
                    encoded_state=encoded_state
                ):
                    return self.runtimeError(action)
            else:
                if not mg.handle_side_step(
                    action, target_cell, self.radius, grid, 
                    invalid_move_reasons= invalid_move_reasons, direction=direction,
                    encoded_state=encoded_state
                ):
                    return self.runtimeError(action)
                
        return None
    
    def step(self, action, render_opp_move=False, print_info=False, act_on_rotated_grid=False):        
        # self.render(print_info=True)
        
        # print(f"action: {action}")
        
        valid_actions = self.generate_valid_moves()
        if len(valid_actions) == 0:
            with open("move_buffer.txt", "w") as file:
                for move in self.move_buffer:
                    file.write(f"{move}")
            win_reward = -10 if self.current_player_to_move() == self.agent_player else 10
            return self.state, win_reward, True, False, self.get_info()
        
        self.move_buffer.append(h.rotate_action(action, self.grid_size) if act_on_rotated_grid else action)
        self.reward = 0 if self.current_player_to_move() != self.agent_player else -1
        
        self.invalid_move_reasons = []
        
        # Execute action["and"] get next state
        # Your environment logic here
        
        # Example reward calculation
        # Prev state no longer used for rewards
        # prev_state = np.copy(self.state)
        terminated = False
        truncated = False

        target_cell = h.Cells.PlayerOne if self.turn % 2 == 0 else h.Cells.PlayerTwo
        enemy_cell = h.Cells.PlayerTwo if self.turn % 2 == 0 else h.Cells.PlayerOne
        
        # print("\nNormal Grid")
        # h.print_hex_grid(self.state)
        # print("Rotated Grid")
        # h.print_hex_grid(self.rotated_state)
        # print(f"action: {action}\nRotated Action: {h.rotate_action(action, self.grid_size)}")
        # print(f"act on rotated grid: {act_on_rotated_grid}")
        # print(f"reward: {self.reward}")
        
        # Perform move on rotated grid as well
        # We need to do normal action first, since we cant rotate invalid moves
        if not act_on_rotated_grid:
            move_result = self.perform_action_on_state(self.state, action, target_cell, enemy_cell, encoded_state=self.encoded_state)
            if move_result is not None:
                return move_result
        else:
            rotated_move_result = self.perform_action_on_state(self.rotated_state, action, enemy_cell, target_cell, encoded_state=self.encoded_rotated_state)
            if rotated_move_result is not None:
                return rotated_move_result
        
        # In theory, should be impossible here, if rotated move failed first
        if not act_on_rotated_grid:
            rotated_move_result = self.perform_action_on_state(self.rotated_state, h.rotate_action(action, self.grid_size), enemy_cell, target_cell, update_info=False, encoded_state=self.encoded_rotated_state)
            if rotated_move_result is not None:
                return move_result
        else:
            move_result = self.perform_action_on_state(self.state, h.rotate_action(action, self.grid_size), target_cell, enemy_cell, encoded_state=self.encoded_state, update_info=False,)
            if move_result is not None:
                return rotated_move_result
            
        # print("after action\nNormal Grid")
        # h.print_hex_grid(self.state)
        # print("Rotated Grid")
        # h.print_hex_grid(self.rotated_state)
        # print("\n\n")
        
        self.turn += 1
        
        p1_count = len(np.where(self.state == h.Cells.PlayerOne)[0])
        p2_count = len(np.where(self.state == h.Cells.PlayerTwo)[0])
        
        if self.grid_size == 5:
            if p1_count <= 3:
                self.winner = h.Cells.PlayerTwo
            elif p2_count <= 3:
                self.winner = h.Cells.PlayerOne
        elif self.grid_size == 3:
            if p1_count <= 1:
                self.winner = h.Cells.PlayerTwo
            elif p2_count <= 1:
                self.winner = h.Cells.PlayerOne
        else:
            raise RuntimeError("Fix code, piece information stuff not being used")
        
        # This dumb code was breaking everything!!
        # if self.piece_information[h.Cells.PlayerOne][h.PieceInformation.OffBoard] >= self.piece_information[h.Cells.PlayerOne][h.PieceInformation.Threshold]:
        #     terminated = True
        #     self.winner = h.Cells.PlayerTwo
        
        # if self.piece_information[h.Cells.PlayerTwo][h.PieceInformation.OffBoard] >= self.piece_information[h.Cells.PlayerTwo][h.PieceInformation.Threshold]:
        #     terminated = True
        #     self.winner = h.Cells.PlayerOne
        
        win_reward = 0
        if self.winner != -1:
            terminated = True
            # Draw, never happens though
            if self.winner == h.Cells.Empty:
                win_reward = -5
            else:    
                win_reward = 10 if self.winner == self.agent_player else -10
        
        # Outdated reward shaping, used PBRS instead in step function
        # self.reward = evaluate_grid(self.state, self.agent_player) + win_reward
        # self.reward = self.compute_reward(prev_state) + win_reward
        self.reward += win_reward
        
        '''
            Code for handling freak accident.
            If you ever get into a position like this, I want you to just skip lil bro's move
        '''
        valid_move_indices = self.generate_valid_moves()
        if len(valid_move_indices) == 0:
            self.turn += 1
              
        if not terminated and self.is_agent and self.current_player_to_move() != self.agent_player:
            return self.perform_opponent_move(render_opp_move=render_opp_move, print_info=print_info)
        else:
            # For one move at a time
            return self.state, self.reward, terminated, truncated, self.get_info()
    
    # For 2 moves at a time
    def perform_opponent_move(self, render_opp_move=False, print_info=False):
        reward = self.reward
        
        if render_opp_move:
            print("after agent move\n")
            self.render(print_info=print_info)
        
        if self.opponent is None:
            # Do random valid move
            valid_actions = self.generate_valid_moves()
            if len(valid_actions) == 0:
                if self.raise_error:
                    raise RuntimeError("got 0 valid moves")
                else:
                    valid_actions.append(self.action_space.sample())
            
            if self.turn == 0 and False:
                _, _, terminated, truncated, _ = self.step({
                    'start': [0, 3],
                    'end': [1, 2],
                    'direction': 4,
                })
            else:
                _, _, terminated, truncated, _ = self.step(random.choice(valid_actions))
            
        else:
            # Do move from model
            if self.agent_player == h.Cells.PlayerOne:
                model_action = self.opponent.model.predict(self.encoded_rotated_state, deterministic=True)[0]
            else:
                model_action = self.opponent.model.predict(self.encoded_state, deterministic=True)[0]
                
            if 'md' in self.opponent.algorithm_name:
                model_action = h.multi_discrete_to_action(model_action, self.grid_size)
            else:
                model_action = h.index_to_action(model_action, self.grid_size)
            
            _, _, terminated, truncated, _ = self.step(model_action, act_on_rotated_grid=self.agent_player == h.Cells.PlayerOne)

        if render_opp_move:
            print(f"opp: reward: {self.reward}")
            print("after opp move\n")
        
        return self.state, self.reward + reward, terminated, truncated, self.get_info()
    
    def get_current_player_marbles(self):
        """Returns coordinates of all marbles for the current player."""
        player_cell = h.Cells.PlayerOne if self.turn % 2 == 0 else h.Cells.PlayerTwo
        return np.argwhere(self.state == player_cell)
    
    def all_marbles(self):
        invalid_cell = h.Cells.Invalid
        return np.argwhere(self.state != invalid_cell)

    def is_valid_position(self, pos):
        """Check if a position is within the board bounds."""
        if h.is_point_in_grid(pos, self.state):
            return True
        return False

    def get_random_move(self):
        """Generates a random valid move for the current player."""
        marbles = self.get_current_player_marbles()
        all_marbles = self.all_marbles()
        if len(marbles) == 0:
            return None  # No marbles left (game over)
        
        # Randomly select a marble and direction
        start = random.choice(marbles)
        direction = random.choice(list(h.Directions))
        end = random.choice(marbles)

        #print(direction)
        
        #dx, dy = h.directions[direction.]
        #dx, dy = h.directions[direction]
        #end = (start[0] + dx, start[1] + dy)
        
        # Ensure the move is valid
        if not self.is_valid_position(end):
            return self.get_random_move()  # Retry if invalid

        return {
            'start': np.array(start),
            'end': np.array(end),
            'direction': direction
        }

    def current_player_to_move(self):
        return h.Cells.PlayerOne if self.turn % 2 == 0 else h.Cells.PlayerTwo
     
    def runtimeError(self, action):
        if (self.raise_error):
            h.print_hex_grid(self.state)
            print(f"action: {action}")
            print(f"action_index: {h.action_to_index(action, grid_size=3)}")
            raise RuntimeError("Got invalid move")

        return self.state, -5, False, False, self.get_info()

def evaluate_grid(grid, active_player):
    grid_size = grid.shape[-1]
    grid_center_pieces = grid[
        h.cell_type_positions[grid_size][h.CellPositionTypes.Center]["rows"],
        h.cell_type_positions[grid_size][h.CellPositionTypes.Center]["cols"],
    ]
    grid_middle_pieces = grid[
        h.cell_type_positions[grid_size][h.CellPositionTypes.Middle]["rows"],
        h.cell_type_positions[grid_size][h.CellPositionTypes.Middle]["cols"],
    ]
    grid_border_pieces = grid[
        h.cell_type_positions[grid_size][h.CellPositionTypes.Border]["rows"],
        h.cell_type_positions[grid_size][h.CellPositionTypes.Border]["cols"],
    ]
    
    grid_center_pieces = {
        h.Cells.PlayerOne: np.sum(grid_center_pieces == h.Cells.PlayerOne),
        h.Cells.PlayerTwo: np.sum(grid_center_pieces == h.Cells.PlayerTwo)
    }
    
    grid_middle_pieces = {
        h.Cells.PlayerOne: np.sum(grid_middle_pieces == h.Cells.PlayerOne),
        h.Cells.PlayerTwo: np.sum(grid_middle_pieces == h.Cells.PlayerTwo)
    }
    
    grid_border_pieces = {
        h.Cells.PlayerOne: np.sum(grid_border_pieces == h.Cells.PlayerOne),
        h.Cells.PlayerTwo: np.sum(grid_border_pieces == h.Cells.PlayerTwo)
    }
    
    direction = 1 if active_player == h.Cells.PlayerOne else -1
    
    center_dominance = (grid_center_pieces[h.Cells.PlayerOne] - grid_center_pieces[h.Cells.PlayerTwo]) * 0.5 * direction
    middle_dominance = (grid_middle_pieces[h.Cells.PlayerOne] - grid_middle_pieces[h.Cells.PlayerTwo]) * 0.35 * direction
    border_dominance = (grid_border_pieces[h.Cells.PlayerOne] - grid_border_pieces[h.Cells.PlayerTwo]) * 0.15 * direction
    
    position_dominance = center_dominance + middle_dominance + border_dominance
    
    pieces_p1 = grid_center_pieces[h.Cells.PlayerOne] + grid_middle_pieces[h.Cells.PlayerOne] + grid_border_pieces[h.Cells.PlayerOne]
    pieces_p2 = grid_center_pieces[h.Cells.PlayerTwo] + grid_middle_pieces[h.Cells.PlayerTwo] + grid_border_pieces[h.Cells.PlayerTwo]
    
    piece_advantage = (pieces_p1 - pieces_p2) * direction
    
    # Calculating attacking and vulnerable pieces
    vulnerable_attacks = 0
    dict_vulnerable_attack = []
    
    for coordinate, value in np.ndenumerate(grid):
        if value == h.Cells.PlayerOne or value == h.Cells.PlayerTwo:
            current_cell = value
            enemy_cell = h.Cells.PlayerOne if current_cell == h.Cells.PlayerTwo else h.Cells.PlayerTwo
            
            for magnitude_direction in range(6):
                cells_in_direction, interacted_with_enemy = collect_cells_in_direction(grid, coordinate, enemy_cell, magnitude_direction)

                if interacted_with_enemy:
                    enemy_start_coordinate = h.add_direction_vector(np.array(cells_in_direction[-1]), magnitude_direction)
                    enemy_cells_in_direction, _ = collect_cells_in_direction(grid, enemy_start_coordinate, current_cell, magnitude_direction)
                    
                    if len(cells_in_direction) > len(enemy_cells_in_direction):
                        enemy_end_coordinate = tuple(enemy_cells_in_direction[-1])
                        is_border_push = enemy_end_coordinate in h.cell_type_positions[grid_size][h.CellPositionTypes.Border]["combined"]
                        is_middle_push = not is_border_push and enemy_end_coordinate in h.cell_type_positions[grid_size][h.CellPositionTypes.Middle]["combined"]
                        
                        if is_border_push:
                            magnitude = 0.5
                        elif is_middle_push:
                            magnitude = 0.35
                        else:
                            magnitude = 0.15
                            
                        direction = 1 if value == active_player else -1
                        vulnerable_attacks += direction * magnitude
                            
                        # dict_vulnerable_attack.append({
                        #     "start": cells_in_direction[0],
                        #     "end": cells_in_direction[-1],
                        #     "direction": direction,
                        # })
                    
    return position_dominance * 0.15 + piece_advantage + vulnerable_attacks * 0.5

def collect_cells_in_direction(grid, start, enemy_cell, direction):
    cells = []
    interacted_with_enemy = False
    
    current_coordinate = np.array(start)
    while True:
        cells.append(current_coordinate)
        next_cell = tuple(h.add_direction_vector(current_coordinate, direction))
        if not h.is_point_in_grid(next_cell, grid) or grid[next_cell] == h.Cells.Empty:
            break
        if grid[next_cell] == enemy_cell:
            interacted_with_enemy = True
            break
        current_coordinate = np.array(next_cell)
    
    return cells, interacted_with_enemy    

# Just to draw independent from env
def draw_box_grid(screen, state, font):
    screen.fill(pygame.Color("white"))
    
    display_info = pygame.display.Info()
    box_size = min(display_info.current_w, display_info.current_h) / (state.shape[0] + 1)
    offset = [box_size, box_size]
    
    h.draw_bordered_square(
        screen,
        "#DAE8FC",
        "#6C8EBF",
        2,
        (0, 0, box_size * (state.shape[1] + 1) + box_size // 2, box_size * (state.shape[0] + 1))
    )
    
    for coordinate, value in np.ndenumerate(state):            
        if value == h.Cells.Invalid:
            color = pygame.Color("#D5E8D4")
            border_color = pygame.Color("#82B366")
        if value == h.Cells.PlayerOne:
            color = pygame.Color("#E1D5E7")
            border_color = pygame.Color("#9673A6")
        if value == h.Cells.PlayerTwo:
            color = pygame.Color("#F8CECC")
            border_color = pygame.Color("#B85450")
        if value == h.Cells.Empty:
            color = pygame.Color("white")
            border_color = pygame.Color("black")
            
        x_offset = box_size // 2 if (coordinate[0] % 2 == 1) else 0
        
        h.draw_bordered_square(
            screen,
            color,
            border_color,
            1,
            (
                coordinate[1] * box_size + x_offset + offset[1], 
                coordinate[0] * box_size + offset[1], 
                box_size, box_size
            )
        )
    
    for index in range(state.shape[0] + 1):
        h.draw_bordered_square(
            screen,
            "#DAE8FC",
            "#6C8EBF",
            2,
            (
                index * box_size, 
                0, 
                box_size, box_size
            )
        )
        
        if index != 0:
            h.draw_centered_text(
                f"{index - 1}",
                screen,
                font,
                pygame.Color("black"),
                index * box_size,
                0,
                box_size,
                box_size
            )
        
        h.draw_bordered_square(
            screen,
            "#DAE8FC",
            "#6C8EBF",
            2,
            (
                0, 
                index * box_size, 
                box_size, box_size
            )
        )
        
        if index != 0:
            h.draw_centered_text(
                f"{index - 1}",
                screen,
                font,
                pygame.Color("black"),
                0,
                index * box_size,
                box_size,
                box_size
            )
    
    pygame.display.flip()
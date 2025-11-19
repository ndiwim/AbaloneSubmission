import pygame
import random
import torch
import numpy as np
import helper as h
import gymnasium as gym
from typing import Tuple
from gymnasium import spaces
from functools import lru_cache


@lru_cache(maxsize=100000)
def generate_valid_moves_cached(is_player_one_to_move: bool, state_bytes: bytes, grid_size):
    valid_moves = []
    
    state = np.frombuffer(state_bytes, dtype=np.int8).reshape((grid_size, grid_size))
    radius = state.shape[-1] // 2
    
    turn = 0 if is_player_one_to_move else 1
    
    target_cell = h.Cells.PlayerOne if turn % 2 == 0 else h.Cells.PlayerTwo
    enemy_cell = h.Cells.PlayerTwo if turn % 2 == 0 else h.Cells.PlayerOne
    
    start = None
    end = None
    
    for index in np.ndindex(state.shape):
        value = state[index]
        if start is None and value == target_cell:
            start = np.array(index)
            end = np.array(index)
        
        if start is not None:
            # First do single block movements
            for direction in range(6):
                action = {
                    'start': start, 
                    'end': end, 
                    'direction': direction
                }
                if check_if_move_is_valid(
                    action, target_cell, enemy_cell, 
                    radius, state, turn, 
                ):
                    valid_moves.append(h.action_to_index(action, grid_size=grid_size))

            # Handle multi block movements
            for no_cells in range(1, 4):
                for selection_direction in range(h.Directions.Right, h.Directions.Left):
                    end = np.array(start)
                    valid_selection = True
                    for _ in range(no_cells):
                        end = h.add_direction_vector(end, selection_direction)
                        if not h.is_point_in_grid(end, state) or state[tuple(end)] != target_cell:
                            valid_selection = False
                            break

                    if valid_selection:
                        for direction in range(6):
                            action = {
                                'start': start, 
                                'end': end, 
                                'direction': direction
                            }
                            
                            if check_if_move_is_valid(
                                action, target_cell, enemy_cell, 
                                radius, state, turn,
                            ):
                                valid_moves.append(h.action_to_index(action, grid_size=grid_size))
            
            start = None
            
    return valid_moves

'''
    Will not return move indices
    if you wanna step where an index isn't allowed, convert it
'''
def generate_valid_moves(turn, state: np.ndarray | torch.Tensor):

    if isinstance(state, torch.Tensor):
        state_buffer = state.numpy().astype(np.int8, copy=False).tobytes()
    else:
        state_buffer = state.astype(np.int8, copy=False).tobytes()
    
    return generate_valid_moves_cached(turn % 2 == 0, state_buffer, state.shape[-1])


def generate_valid_moves_batch_md(obs_batch):
    grid_size = obs_batch.shape[-1]
    batch_size = obs_batch.shape[0]
    traversable_dict = {}
    
    for batch_number in range(batch_size):
        traversable_dict[batch_number] = {}        
        valid_moves = h.turn_indices_to_actions(generate_valid_moves(0, obs_batch[batch_number].argmax(dim=0)), obs_batch.shape[-1])
        
        for move in valid_moves:
            start_index = h.coordinate_to_index(move['start'], grid_size)
            end_index = h.coordinate_to_index(move['end'], grid_size)
            selection_magnitude_direction = h.multi_discrete_determine_selection_magnitude(start_index, end_index, grid_size)
            is_same_cell = start_index == end_index
            
            if start_index not in traversable_dict[batch_number]:
                traversable_dict[batch_number][start_index] = {}
                
            if selection_magnitude_direction not in traversable_dict[batch_number][start_index]:
                if is_same_cell:
                    for i in range(6):
                        traversable_dict[batch_number][start_index][i] = []
                else:
                    traversable_dict[batch_number][start_index][selection_magnitude_direction] = []
            
            if is_same_cell:
                for i in range(6):
                    traversable_dict[batch_number][start_index][i].append(move['direction'])
            else:
                traversable_dict[batch_number][start_index][selection_magnitude_direction].append(move['direction'])
            
    return traversable_dict
        

def check_if_move_is_valid(action, target_cell, enemy_cell, radius, state, turn, invalid_move_reasons=None, piece_information=None):
    if invalid_start_end(action, state, invalid_move_reasons=invalid_move_reasons):
        return False
    
    direction, magnitude = h.determine_direction(action["start"], action["end"], radius)
    
    if check_if_is_invalid_direction(
        action, direction, magnitude, radius, 
        invalid_move_reasons=invalid_move_reasons, move_pieces_on_board=False
    ):
        return False
    
    if not check_selected_cells_are_yours(
        action, direction, target_cell, state, turn, 
        invalid_move_reasons=invalid_move_reasons, move_pieces_on_board=False
    ):
        return False
    
    is_single_block_move, valid_move = handle_single_block_movement(
        action, radius, state, magnitude=magnitude, 
        move_pieces_on_board=False, invalid_move_reasons=invalid_move_reasons
    )
    
    if is_single_block_move and not valid_move:
        return False
    
    if not is_single_block_move:
        if is_push_move(direction, action) and not handle_push_logic(
            action, target_cell, enemy_cell, state, radius, invalid_move_reasons=invalid_move_reasons,
            magnitude=magnitude, direction=direction, move_pieces_on_board=False, piece_information=piece_information
        ):
            return False
        
        if not is_push_move(direction, action) and not handle_side_step(
            action, target_cell, radius, state, invalid_move_reasons=invalid_move_reasons,
            direction=direction, move_pieces_on_board=False
            
        ):
            return False
    
    return True        


def check_if_is_invalid_direction(action, direction, magnitude, radius, move_pieces_on_board=True, invalid_move_reasons=None) -> int:
    if invalid_move_reasons is None:
        invalid_move_reasons = []
    
    # Direction is invalid and magnitude is not 1, or magnitude is more than 3
    if (direction == -1 and magnitude != 1 or magnitude > 3):
        if direction == -1 and move_pieces_on_board:
            invalid_move_reasons.append("Direction between point (" \
                + h.convert_odd_row_to_cube(action['start'], radius, to_string=True) \
                + ") and point (" \
                + h.convert_odd_row_to_cube(action['end'], radius, to_string=True) \
                + ") is invalid")
        if magnitude > 3 and move_pieces_on_board:
            invalid_move_reasons.append("Magnitude of " + str(magnitude) + " is greater than 3.")
        
    return (direction == -1 and magnitude != 1 or magnitude > 3)


def handle_single_block_movement(action, radius, state, invalid_move_reasons=None, magnitude = None, move_pieces_on_board=True, encoded_state=None) -> Tuple[bool, bool]:
    if invalid_move_reasons is None:
        invalid_move_reasons = []
    
    if magnitude is None:
        _, magnitude = h.determine_direction(action["start"], action["end"], radius)
    
    if magnitude == 1:    
        next_cell = h.add_direction_vector(action['start'], action['direction'])
        if not h.is_point_in_grid(next_cell, state):
            if move_pieces_on_board:
                invalid_move_reasons.append(f"Moving cell {action['start']} in the direction {action['direction']} is off the grid")
            return True, False
        
        elif state[tuple(next_cell)] != h.Cells.Empty:
            if move_pieces_on_board:
                invalid_move_reasons.append(f"Moving cell {action['start']} in the direction {action['direction']} is not an empty cell.")
            return True, False
        
        else:
            if move_pieces_on_board:
                target_cell = state[tuple(action['start'])]
                
                start_tuple = tuple(action['start'])
                next_tuple = tuple(next_cell)
                
                old_start_cell = state[start_tuple]
                old_next_cell = state[next_tuple]
                
                state[start_tuple] = h.Cells.Empty
                state[next_tuple] = target_cell
                
                
                if encoded_state is not None:
                    encoded_state[h.Cells.Empty.value, start_tuple[0], start_tuple[1]] = True
                    encoded_state[target_cell, next_tuple[0], next_tuple[1]] = True
                    
                    encoded_state[old_start_cell, start_tuple[0], start_tuple[1]] = False
                    encoded_state[old_next_cell, next_tuple[0], next_tuple[1]] = False
                    
            return True, True

    return False, True
   
    
def handle_push_logic(action, target_cell, enemy_cell, state, radius, invalid_move_reasons=None, piece_information=None, magnitude = None, direction = None, move_pieces_on_board=True, encoded_state=None) -> bool:
    if invalid_move_reasons is None:
        invalid_move_reasons = []
        
    if direction is None or magnitude is None:
        direction, magnitude = h.determine_direction(action["start"], action["end"], radius)

    pickup_point = action["start"] if action['direction'] == direction else action["end"]
    current_point = action["end"] if action['direction'] == direction else action["start"]
    next_point = h.add_direction_vector(current_point, action['direction'])
    
    if not h.is_point_in_grid(next_point, state) or state[tuple(next_point)] == target_cell:
        if move_pieces_on_board:
            if not h.is_point_in_grid(next_point, state):
                invalid_move_reasons.append(
                    "Grid at position (" + h.convert_odd_row_to_cube(next_point, radius, to_string=True) + ") " \
                    + "is not a valid grid position"
                )
            
            elif state[tuple(next_point)] == target_cell:
                invalid_move_reasons.append(
                    "Grid at position (" + h.convert_odd_row_to_cube(next_point, radius, to_string=True) + ") " \
                    + "is the target cell. Cannot make a push move into yourself"
                )
        
        return False
    
    if (state[tuple(next_point)] == h.Cells.Empty):
        if move_pieces_on_board:
            pickup_tuple = tuple(pickup_point)
            next_point_tuple = tuple(next_point)
            
            pickup_cell = state[pickup_tuple]
            next_cell = state[next_point_tuple]
            
            state[pickup_tuple] = h.Cells.Empty
            state[next_point_tuple] = target_cell
            
            if encoded_state is not None:
                encoded_state[h.Cells.Empty.value, pickup_tuple[0], pickup_tuple[1]] = True
                encoded_state[target_cell, next_point_tuple[0], next_point_tuple[1]] = True
                
                encoded_state[pickup_cell, pickup_tuple[0], pickup_tuple[1]] = False
                encoded_state[next_cell, next_point_tuple[0], next_point_tuple[1]] = False
            
        return True
    
    target_group = []
    is_push_off_board = False
    while len(target_group) < magnitude:
        target_group.append(next_point)
        
        next_point = h.add_direction_vector(next_point, action['direction'])
        
        if not h.is_point_in_grid(next_point, state) or state[tuple(next_point)] == h.Cells.Empty:
            if move_pieces_on_board:
                is_push_off_board = not h.is_point_in_grid(next_point, state)
            
                if is_push_off_board and piece_information is not None:
                    piece_information[enemy_cell][h.PieceInformation.OnBoard] -= 1
                    piece_information[enemy_cell][h.PieceInformation.OffBoard] += 1
            break
    
    if len(target_group) < magnitude:
        if move_pieces_on_board:
            starting_point = action["start"] if action['direction'] == direction else action["end"]
            ending_point = action["end"] if action['direction'] == direction else action["start"]
            starting_enemy_point = h.add_direction_vector(ending_point, action['direction'])
            
            start_tuple = tuple(starting_point)
            enemy_tuple = tuple(starting_enemy_point)
            
            old_start_cell = state[start_tuple]
            old_enemy_cell = state[enemy_tuple]
            
            state[start_tuple] = h.Cells.Empty
            state[enemy_tuple] = target_cell
            
            if encoded_state is not None:
                encoded_state[h.Cells.Empty.value, start_tuple[0], start_tuple[1]] = True
                encoded_state[target_cell, enemy_tuple[0], enemy_tuple[1]] = True
                
                encoded_state[old_start_cell, start_tuple[0], start_tuple[1]] = False
                encoded_state[old_enemy_cell, enemy_tuple[0], enemy_tuple[1]] = False
            
            if not is_push_off_board:
                next_tuple = tuple(next_point)
                old_next_cell = state[next_tuple]
                
                state[next_tuple] = enemy_cell
                
                if encoded_state is not None:
                    encoded_state[enemy_cell, next_tuple[0], next_tuple[1]] = True
                    encoded_state[old_next_cell, next_tuple[0], next_tuple[1]] = False
                
        return True
    else:
        if move_pieces_on_board:
            invalid_move_reasons.append("Push move of magnitude " + str(magnitude) + " is insufficient due to more enemy marbles")
        
        return False


def handle_side_step(action, target_cell, radius, state, invalid_move_reasons=None, direction=None, move_pieces_on_board=True, encoded_state=None) -> bool:
    if invalid_move_reasons is None:
        invalid_move_reasons = []
    
    if direction is None:
        direction, _ = h.determine_direction(action["start"], action["end"], radius)
    
    current_point = action["start"]
    # Think of better name
    target_group = []
    starting_group = []
    
    while True:
        if (state[tuple(current_point)] != target_cell):
            if move_pieces_on_board:
                invalid_move_reasons.append(
                    "Grid at position (" + h.convert_odd_row_to_cube(current_point, radius, to_string=True) + ") = " \
                    + str(state[tuple(current_point)]) + " "\
                    + "is not the target cell " + str(target_cell)
                )
            
            return False

        next_point = h.add_direction_vector(current_point, action["direction"])
        
        '''
            Invalid move if target cells are:
            - Off the grid
            - Not Empty
            - In invalid grid position
            - Handing the push logic in ---if (action["direction"] % 3 == direction % 3):---
        '''
        if (not h.is_point_in_grid(next_point, state) or state[tuple(next_point)] != h.Cells.Empty):
            
            if not h.is_point_in_grid(next_point, state):
                if move_pieces_on_board:
                    invalid_move_reasons.append(
                        "Grid at position (" + h.convert_odd_row_to_cube(next_point, radius, to_string=True) + ") " \
                        + "is not a valid grid position / grid is marked invalid there."
                    )
                
            elif state[tuple(next_point)] != h.Cells.Empty:
                if move_pieces_on_board:
                    invalid_move_reasons.append(
                        "Action is a side step move, and grid at position (" + h.convert_odd_row_to_cube(next_point, radius, to_string=True) + ") " \
                        + "is not empty"
                    )
            
            return False
        
        target_group.append(tuple(next_point))
        starting_group.append(tuple(current_point))
        
        if (np.equal(current_point, action["end"]).all()):
            break
            
        current_point = h.add_direction_vector(current_point, direction)

    if move_pieces_on_board:
        for starting_point, ending_point in zip(starting_group, target_group):
            
            old_start_cell = state[starting_point]
            old_end_cell = state[ending_point]
            
            state[starting_point] = h.Cells.Empty
            state[ending_point] = target_cell
            
            if encoded_state is not None:
                encoded_state[h.Cells.Empty.value, starting_point[0], starting_point[1]] = True
                encoded_state[target_cell, ending_point[0], ending_point[1]] = True
                
                encoded_state[old_start_cell, starting_point[0], starting_point[1]] = False
                encoded_state[old_end_cell, ending_point[0], ending_point[1]] = False
            
    return True


def check_selected_cells_are_yours(action, direction, target_cell, state, turn, invalid_move_reasons=None, move_pieces_on_board=True) -> bool:
    if invalid_move_reasons is None:
        invalid_move_reasons = []
    
    # If not all selected cells are yours
    if direction != -1:      
        stop_next = False
        current_point = action["start"]
        while (True):
            # Todo, enhance errors sis
            if not h.is_point_in_grid(current_point, state):
                if move_pieces_on_board:
                    invalid_move_reasons.append("Path between start and end is not entirely on the board")
                return False
            
            elif state[tuple(current_point)] != target_cell:
                if move_pieces_on_board:
                    invalid_move_reasons.append(f"Path between start and end is not entirely player {(turn % 2 + 1)}")
                return False
            
            else:
                current_point = h.add_direction_vector(current_point, direction)
            
            if (stop_next):
                break
            
            stop_next = np.equal(current_point, action["end"]).all()  
    
    return True


def is_push_move(direction, action):
    return direction >= 0 and action["direction"] % 3 == direction % 3


def invalid_start_end(action, state, invalid_move_reasons=None):
    if invalid_move_reasons == None:
        invalid_move_reasons = []
    if not h.is_point_in_grid(action['start'], state) or not h.is_point_in_grid(action['end'], state):
        invalid_move_reasons.append(f"Action includes invalid grid squares {action}")
        return True
    return False
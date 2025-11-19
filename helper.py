import os
import re
import ast
import torch
import random
import pygame
import copy
import numpy as np
import pandas as pd
from enum import IntEnum, StrEnum, Enum
from numbers import Integral
from tensorboard.backend.event_processing import event_accumulator

# Changed default device to cpu since both A2C and PPO are meant to be used on the CPU...
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"

class Directions(IntEnum):
    UpLeft = 0
    UpRight = 1
    Right = 2
    DownRight = 3
    DownLeft = 4
    Left = 5
    
class Cells(IntEnum):
    Invalid = 0
    PlayerOne = 1
    PlayerTwo = 2
    Empty = 3
    
class PieceInformation(IntEnum):
    OnBoard = 0
    OffBoard = 1
    Threshold = 2
    
class RenderModes(StrEnum):
    Human = 'human'
    Grid = 'grid'
    BoxPygame = 'box_pygame'

directions = {
    0: {  # Even rows
        Directions.UpLeft: np.array([-1, -1]),
        Directions.UpRight: np.array([-1, 0]),
        Directions.Right: np.array([0, 1]),
        Directions.DownRight: np.array([1, 0]),
        Directions.DownLeft: np.array([1, -1]),
        Directions.Left: np.array([0, -1])
    },
    1: {  # Odd rows
        Directions.UpLeft: np.array([-1, 0]),
        Directions.UpRight: np.array([-1, 1]),
        Directions.Right: np.array([0, 1]),
        Directions.DownRight: np.array([1, 1]),
        Directions.DownLeft: np.array([1, 0]),
        Directions.Left: np.array([0, -1])
    }
}

class CellPositionTypes(IntEnum):
    Center = 0
    Middle = 1
    Border = 2

cell_type_positions = {
    # 2x2x2 grid
    3: {
        CellPositionTypes.Center: {
            "rows": [1],
            "cols": [1],
            "combined": [(1, 1)]
        },
        CellPositionTypes.Middle: {"rows": [], "cols": [], "combined": []},
        CellPositionTypes.Border: {
            "rows": [0, 0, 1, 1, 2, 2],
            "cols": [1, 2, 0, 2, 1, 2],
            "combined": [
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 2),
                (2, 1),
                (2, 2),
            ]
        },
    },
    # 3x3x3 grid
    5: {
        CellPositionTypes.Center: {
            "rows": [2],
            "cols": [2],
            "combined": [(2, 2)]
        },
        CellPositionTypes.Middle: {
            "rows": [1, 1, 2, 2, 3, 3],
            "cols": [1, 2, 1, 3, 1, 2],
            "combined": [
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 1),
                (3, 2)
            ]
        },
        CellPositionTypes.Border: {
            "rows": [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
            "cols": [1, 2, 3, 0, 3, 0, 4, 0, 3, 1, 2, 3],
            "combined": [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 3),
                (2, 0),
                (2, 4),
                (3, 0),
                (3, 3),
                (4, 1),
                (4, 2),
                (4, 3)
            ]
        }
    }
    # Others aren't implemented yet
}

cube_directions = {
    Directions.UpLeft: np.array([0, 1, -1]),
    Directions.UpRight: np.array([1, 0, -1]),
    Directions.Right: np.array([1, -1, 0]),
    Directions.DownRight: np.array([0, -1, 1]),
    Directions.DownLeft: np.array([-1, 0, 1]),
    Directions.Left: np.array([-1, 1, 0])
}

class LearnLog():
    def __init__(self, enabled=False, interval=0, no_rounds=10):
        self.counter = 0
        self.enabled=enabled
        self.interval=interval
        self.interval_step = None
        self.no_rounds = no_rounds
        
        self.history = {}
        
class OpponentClass():
    def __init__(self, model_path, algorithm_name, raise_error):
        self.model_path = model_path
        self.algorithm_name = algorithm_name
        self.raise_error = raise_error
        self.model = None
        
def make_hex_grid(size, set_initial_position=True, piece_information=None):
    size = 2 * size -1
    grid = np.full(shape=(size, size), fill_value=Cells.Empty)
    if (set_initial_position):
        set_initial_player_configuration(grid, piece_information)
    mark_invalid_cells(grid)
    return grid

def get_invalid_cells_in_grid(grid_size):
    invalid_starting = []
    invalid_ending = []
    
    radius = grid_size // 2
    for row_index in range(grid_size):
        axial_row = row_index - radius
        no_cols_in_row = 2 * radius + 1 - abs(axial_row)
        if radius % 2 == 0:
            starting_index = abs(axial_row) // 2
        else:
            starting_index = int(np.ceil(abs(axial_row) / 2))
            
        invalid_starting.append(starting_index)
        invalid_ending.append(grid_size - (starting_index + no_cols_in_row))
        
    return invalid_starting, invalid_ending

def mark_invalid_cells(grid):
    radius = grid.shape[-1] // 2
    for row_index, row in enumerate(grid):
        axial_row = row_index - radius
        no_cols_in_row = 2 * radius + 1 - abs(axial_row)
        if radius % 2 == 0:
            starting_index = abs(axial_row) // 2
        else:
            starting_index = int(np.ceil(abs(axial_row) / 2))
        row[: starting_index] = Cells.Invalid
        row[starting_index + no_cols_in_row: ] = Cells.Invalid

def get_piece_threshold(grid_size):
    match grid_size:
        case 3: return 1
        case 5: return 2
        case 7: return 5
        case 9: return 6
        case 11: return 7
        case _: raise NotImplementedError("Default configurations for grid sizes 3, 5, 7, 9 and 11")

def get_initial_no_pieces(grid_size):
    match grid_size:
        case 3: return 2
        case 5: return 5
        case 7: return 11
        case 9: return 14
        case 11: return 29
        case _: raise NotImplementedError("Default configurations for grid sizes 3, 5, 7, 9 and 11")

def set_initial_player_configuration(grid, piece_information):
    grid_size = grid.shape[-1]
    piece_threshold = get_piece_threshold(grid_size)
    match grid_size:
        case 3: 
            grid[0, 1:3] = Cells.PlayerOne
            grid[2, 1:3] = Cells.PlayerTwo
            

            if piece_information is not None:           
                piece_information[Cells.PlayerOne] = [2, 0, piece_threshold]
                piece_information[Cells.PlayerTwo] = [2, 0, piece_threshold]
        case 5: 
            grid[0, :] = Cells.PlayerOne
            grid[1, 1:3] = Cells.PlayerOne
            
            grid[-2, 1:3] = Cells.PlayerTwo
            grid[-1, :] = Cells.PlayerTwo
            
            if piece_information is not None:
                piece_information[Cells.PlayerOne] = [5, 0, piece_threshold]
                piece_information[Cells.PlayerTwo] = [5, 0, piece_threshold]
        case 7: 
            grid[:2, :] = Cells.PlayerOne
            grid[2, 3:5] = Cells.PlayerOne
            
            grid[-2:, :] = Cells.PlayerTwo
            grid[-3, 3:5] = Cells.PlayerTwo
            
            if piece_information is not None:
                piece_information[Cells.PlayerOne] = [11, 0, piece_threshold]
                piece_information[Cells.PlayerTwo] = [11, 0, piece_threshold]
        case 9: 
            grid[:2, :] = Cells.PlayerOne
            grid[2, 3:6] = Cells.PlayerOne
            
            grid[-2:, :] = Cells.PlayerTwo
            grid[-3, 3:6] = Cells.PlayerTwo
            
            if piece_information is not None:
                piece_information[Cells.PlayerOne] = [14, 0, piece_threshold]
                piece_information[Cells.PlayerTwo] = [14, 0, piece_threshold]
        case 11: 
            grid[:3, :] = Cells.PlayerOne
            grid[3, 3:8] = Cells.PlayerOne
            
            grid[-3:, :] = Cells.PlayerTwo
            grid[-4, 3:8] = Cells.PlayerTwo
            
            if piece_information is not None:
                piece_information[Cells.PlayerOne] = [29, 0, piece_threshold]
                piece_information[Cells.PlayerTwo] = [29, 0, piece_threshold]
        case _: raise NotImplementedError("Default configurations for grid sizes 3, 5, 7, 9 and 11")
        
def print_hex_grid(grid):
    size = grid.shape[-1]    
    radius = grid.shape[-1] // 2
    number_of_hashtags = (size + 1) // 2 + 1
    number_of_spaces = radius + 1 + radius % 2
    
    line = ''
    for _ in range(number_of_spaces):
        line += ' '
    for _ in range(number_of_hashtags):
        line += '# '
    print(line)
    
    for row_index, row in enumerate(grid):
        axial_row = row_index - radius
        no_cols_in_row = 2 * radius + 1 - abs(axial_row)
        if radius % 2 == 0:
            starting_index = abs(axial_row) // 2
        else:
            starting_index = int(np.ceil(abs(axial_row) / 2))
        
        line = ''
        if (row_index % 2 == 1):
            line += ' '
            
        if (row_index == 4):
            t = 1
            
        no_hashtags_drawn = 0
        for col_index, col in enumerate(row):
            if col_index == starting_index or col_index == starting_index + no_cols_in_row:
                line += '# '
                no_hashtags_drawn += 1
            line += col.astype(str)
            if (col_index != size - 1):
                line += ' '
            elif no_hashtags_drawn < 2:
                line += ' #'
        
        line = line.replace('0', ' ')
        line = line.replace('3', '-')
        print(line)
            
    line = ''
    for _ in range(number_of_spaces):
        line += ' '
    for _ in range(number_of_hashtags):
        line += '# '
    print(line)
        
def convert_odd_row_to_cube(odd_row_coordinate, radius, to_string=False):
    axial_row = odd_row_coordinate[0] - radius
    
    if radius % 2 == 0:
        starting_index = abs(axial_row) // 2
    else:
        starting_index = int(np.ceil(abs(axial_row) / 2))
    
    if odd_row_coordinate[0] < radius:
        center = starting_index + odd_row_coordinate[0]
    else: 
        center = starting_index + radius
    
    x = odd_row_coordinate[1] - center
    z = axial_row
    y = -x-z
    return np.array([x, y, z]) if not to_string else ','.join(np.array([x, y, z]).astype(str))

def determine_direction(starting_point, ending_point, radius):
    cube_starting_point = convert_odd_row_to_cube(starting_point, radius)
    cube_ending_point = convert_odd_row_to_cube(ending_point, radius)
    
    direction_vector = (cube_ending_point - cube_starting_point).astype(np.float64)
    
    magnitude = np.max(direction_vector)

    if magnitude == 0:
        return -1, 1

    direction_vector /= magnitude
    direction_vector = direction_vector.astype(np.int64)
    magnitude += 1
    
    for direction in range(6):
        if np.equal(cube_directions[direction], direction_vector).all():
            return direction, magnitude

    return -1, magnitude

def is_point_in_grid(point, grid):
    return point[0] >= 0 \
        and point[1] >= 0 \
        and point[0] < grid.shape[0] \
        and point[1] < grid.shape[1] \
        and grid[tuple(point)] != Cells.Invalid
        
def add_direction_vector(point, direction):
    return point + directions[point[0] % 2][direction]

# Todo, you can fix the width, and make square overlap
def draw_bordered_square(screen, fill_color, border_color, width, rect):
    # Fill Square
    pygame.draw.rect(
        screen, 
        fill_color, 
        rect
    )
    
    # Border Square
    pygame.draw.rect(
        screen, 
        border_color, 
        rect,
        1
    )
    
def draw_centered_text(text, screen, font, color, x, y, length, height):
    text_surface = font.render(text, True, color)
    
    screen.blit(text_surface, (
        x + length / 2 - text_surface.get_width() / 2,
        y + height / 2 - text_surface.get_height() / 2,
    ))
    
def action_to_index(action, grid_size):
    return  (action['start'][0] * grid_size + action['start'][1]) * grid_size ** 2 * 6 +\
            (action['end'][0] * grid_size + action['end'][1]) * 6 +\
            action['direction']
    
# Mixed-radix encoding        
def index_to_action(index, grid_size):
    start, rem = divmod(index, grid_size ** 2 * 6)
    end, direction = divmod(rem, 6)
    return {
        'start': np.array([start // grid_size, start % grid_size], dtype=np.int64),
        'end': np.array([end // grid_size, end % grid_size], dtype=np.int64),
        'direction': direction
    }
    
def multi_discrete_determine_selection_magnitude(start_index, end_index, grid_size):
    start_cell = np.array(index_to_coordinate(start_index, grid_size))
    end_cell = np.array(index_to_coordinate(end_index, grid_size))
    
    magnitude = np.abs(start_cell - end_cell).max()
    
    if magnitude == 0:
        return 0
    
    # Horizontal
    if start_cell[0] == end_cell[0]:
        direction = Directions.Right if end_cell[1] > start_cell[1] else Directions.Left
    # Upwards
    elif end_cell[0] < start_cell[0]:
        if start_cell[0] % 2 == 0:
            direction = Directions.UpRight if end_cell[1] >= start_cell[1] else Directions.UpLeft
        else:
            direction = Directions.UpRight if end_cell[1] > start_cell[1] else Directions.UpLeft
    # Downwards
    else:
        if start_cell[0] % 2 == 0:
            direction = Directions.DownRight if end_cell[1] >= start_cell[1] else Directions.DownLeft
        else:
            direction = Directions.DownRight if end_cell[1] > start_cell[1] else Directions.DownLeft
            
    return 6 * magnitude + direction
    
def multi_discrete_get_end_index(start_index, selection_direction_magnitude, grid_size):
    start_cell = index_to_coordinate(start_index, grid_size)
    
    selection_direction = selection_direction_magnitude % 6
    magnitude = selection_direction_magnitude // 6
    
    end_cell = start_cell.copy()
    for _ in range(magnitude):
        end_cell = add_direction_vector(end_cell, selection_direction)
        
    return coordinate_to_index(end_cell, grid_size)
    
def multi_discrete_to_action(action, grid_size):
    start_cell_index = 0
    selection_magnitude_index = 1
    move_direction_index = 2
    
    selection_direction = action[selection_magnitude_index] % 6
    magnitude = action[selection_magnitude_index] // 6
    
    start = index_to_coordinate(action[start_cell_index], grid_size)
    end = None
    
    match (magnitude):
        case 0: end = start.copy()
        case 1: end = add_direction_vector(start, selection_direction)
        case 2: 
            end = add_direction_vector(start, selection_direction)
            end = add_direction_vector(end, selection_direction)

    if end is None:
        raise RuntimeError("mutli discrete convert, why am I none!??")

    return {
        'start': start,
        'end': end,
        'direction': action[move_direction_index],
    }
    
def multi_discrete_to_index(action, grid_size):
    return action_to_index(action, grid_size)   
    
def dict_action_to_string(action):
    return f"start: [{action['start'].squeeze()[0]}, {action['start'].squeeze()[1]}]\nend:[{action['end'].squeeze()[0]}, {action['end'].squeeze()[1]}]\ndirection: {action['direction']}\n"
    
def multi_discrete_indices_to_action(action, grid_size):
    return {
        "start": index_to_coordinate(action[0], grid_size),
        "end": index_to_coordinate(action[1], grid_size),
        "direction": action[2]
    }
    
def index_to_coordinate(index, grid_size):
    return [index // grid_size, index % grid_size]

def coordinate_to_index(coordinate, grid_size):
    return coordinate[0] * grid_size + coordinate[1]

def turn_actions_to_indices(valid_moves, grid_size):
    indices = []
    for action in valid_moves:
        indices.append(action_to_index(action, grid_size))
    
    mask = np.isin(np.arange(grid_size ** 4 * 6), indices)
        
    return indices, mask

def turn_indices_to_actions(valid_move_indices, grid_size):
    actions = []
    for action_index in valid_move_indices:
        actions.append(index_to_action(action_index, grid_size))
    
    return actions

def set_seeds(seed=42):
    random.seed(seed)          # Python RNG
    np.random.seed(seed)       # NumPy RNG
    torch.manual_seed(seed)    # Torch RNG (CPU)
    torch.cuda.manual_seed(seed)  # Torch RNG (GPU, if used)
    torch.backends.cudnn.deterministic = True  # make CuDNN deterministic
    torch.backends.cudnn.benchmark = False     # disable heuristics
    
def parse_move_buffer():
    with open("./move_buffer.txt", "r") as f:
        raw = f.read()

    cleaned = re.sub(r"array\((\[.*?\])\)", r"\1", raw)
    cleaned = re.sub(r"np\.int64\((\d+)\)", r"\1", cleaned)
    cleaned = re.sub(r'}{', '},{', cleaned)
    return ast.literal_eval(f"[{cleaned}]")

def grid_to_mlp(observations, flatten=True):
    # Its a batch
    if len(observations.shape) == 2:
        return one_hot_encode_grid(observations, flatten=flatten)
    
    one_hot_encoded_grids = []
    for batch_index in range(observations.shape[-1]):
        one_hot_encoded_grids.append(torch.tensor(one_hot_encode_grid(observations[batch_index, :, :], flatten=flatten)))
    stacked = torch.stack(one_hot_encoded_grids)
    return stacked

def one_hot_encode_grid(grid, flatten=True):
    # grid_size = grid.shape[-1]
    # encoded_grid = np.zeros((4, grid_size, grid_size), dtype=np.int8)
    # for piece_type in range(4):    
    #     rows, cols = (grid == piece_type).nonzero()
    #     encoded_grid[piece_type, rows, cols] = 1
    # return encoded_grid.flatten() if flatten else encoded_grid
    
    encoded = np.eye(4, dtype=np.int8)[grid]  # (H, W, 4)
    encoded = encoded.transpose(2, 0, 1)      # (4, H, W)
    return encoded.flatten() if flatten else encoded
    
    # torch_encoded_grid = torch.tensor(encoded_grid, dtype=torch.int8)
    # return torch_encoded_grid.flatten() if flatten else torch_encoded_grid

def get_model_list(algorithm_name, hex_grid_size=2):
    train_path = get_train_path(algorithm_name, hex_grid_size=hex_grid_size)
    
    if not os.path.isdir(train_path):
        raise RuntimeError(f"models in folder {train_path} dont exist")

    pattern = rf'({algorithm_name})_(\d+)\.zip'
    models = []
    model_numbers = []
    for filename in os.listdir(train_path):
        match = re.match(pattern, filename)
        if match:
            model_numbers.append(int(match.group(2)))
            
    model_numbers.sort()
    for model_number in model_numbers:
        models.append(os.path.join(train_path, f"{algorithm_name}_{model_number}.zip"))
    
    return models


def get_train_path(algorithm_name, hex_grid_size=2):
    dimensions_str = f"{hex_grid_size}x{hex_grid_size}x{hex_grid_size}"
    
    return f"./models/{algorithm_name}/{dimensions_str}/train"

def get_best_model(algorithm_name, hex_grid_size=2, train_path=None):
    if train_path is None:
        train_path = get_train_path(algorithm_name, hex_grid_size=hex_grid_size)
    
    if not os.path.isdir(train_path):
        os.makedirs(train_path, exist_ok=True)
        # raise RuntimeError(f"models in folder {train_path} dont exist")

    pattern = rf'({algorithm_name})_(\d+)\.zip'
    best_model = None
    for filename in os.listdir(train_path):
        match = re.match(pattern, filename)
        if match:
            model_number = int(match.group(2))
            best_model = model_number if best_model is None or model_number > best_model else best_model
            
    if best_model is None:
        # raise RuntimeError(f"Couldn't find best {algorithm_name} model in folder {train_path}")
        return None, -1

    return os.path.join(train_path, f"{algorithm_name}_{best_model}.zip"), best_model

def get_model(algorithm_name, hex_grid_size=2):
    dimensions_str = f"{hex_grid_size}x{hex_grid_size}x{hex_grid_size}"
    
    return f"./models/{algorithm_name}/{dimensions_str}"
    
def get_logs(algorithm_name, hex_grid_size=2):
    dimensions_str = f"{hex_grid_size}x{hex_grid_size}x{hex_grid_size}"
    
    return f"./models/{algorithm_name}/logs/{dimensions_str}"
   
# def flip_actions()

def rotate_grid(grid, in_place=True, rotate_players=True):
    grid_in_use = grid if in_place else grid.copy()
    match(grid.shape[-1]):
        case 3: 
            grid_in_use[[0, -1], 1:] = grid_in_use[[-1, 0], -1:0:-1]
            grid_in_use[1, :] = grid_in_use[1, ::-1]
        case 5:
            grid_in_use[[0, -1], 1:-1] = grid_in_use[[-1, 0], -2:0:-1]
            grid_in_use[[1, -2], :-1] = grid_in_use[[-2, 1], -2::-1]
            grid_in_use[2, :] = grid_in_use[2, ::-1]
        case _: raise NotImplementedError("Yet to implement flip for hex_grids greater than 3")
        
    if rotate_players:
        p1_indices = np.where(grid_in_use == Cells.PlayerOne)
        p2_indices = np.where(grid_in_use == Cells.PlayerTwo)
        
        grid_in_use[p1_indices] = Cells.PlayerTwo
        grid_in_use[p2_indices] = Cells.PlayerOne
    return grid_in_use

def rotate_action(action, grid_size):
    is_index_action = isinstance(action, Integral)
    if is_index_action:
        action_dict = index_to_action(action, grid_size=grid_size)
    else:
        action_dict = copy.deepcopy(action)
    
    start_coordinate = np.array(action_dict['start']) if not isinstance(action_dict['start'], np.ndarray) else action_dict['start']
    end_coordinate = np.array(action_dict['end']) if not isinstance(action_dict['end'], np.ndarray) else action_dict['end']
    
    invalid_starting, invalid_ending = get_invalid_cells_in_grid(grid_size)
    
    action_dict['start'] = grid_size - start_coordinate - 1
    action_dict['end'] = grid_size - end_coordinate - 1
    
    action_dict['start'][1] += invalid_starting[start_coordinate[0]] - invalid_ending[start_coordinate[0]]
    action_dict['end'][1] += invalid_starting[end_coordinate[0]] - invalid_ending[end_coordinate[0]]
    
    action_dict['direction'] = (action_dict['direction'] + 3) % 6
    
    if is_index_action:
        return action_to_index(action_dict, grid_size)
    else:
        return action_dict
    
def extract_logs(log_path):
    event_file = os.path.join(log_path, os.listdir(log_path)[0])
    
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()  # Load the file

    # Collect all scalars
    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value, e.wall_time) for e in events]

    # Flatten to a DataFrame
    rows = []
    for tag, values in scalars.items():
        for step, value, wall_time in values:
            rows.append([tag, step, value, wall_time])

    df = pd.DataFrame(rows, columns=["tag", "step", "value", "wall_time"])
    
    pivoted = df.pivot_table(
        index="step",  # rows
        columns="tag",      # columns
        values="value",     # values
        aggfunc="first"     # if duplicates, just take first
    ).reset_index()

    pivoted.to_csv(os.path.join(log_path, "logs.csv"), index=False)
    
def get_opp_player(player):
    return Cells.PlayerTwo if player == Cells.PlayerTwo else Cells.PlayerOne
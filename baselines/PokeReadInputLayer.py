from math import floor

import numpy as np

def make_reward_channel(r_val, w, h, col_steps):
    col_steps
    row = floor(r_val / (h * col_steps))
    memory = np.zeros(shape=(h, w), dtype=np.uint8)
    memory[:, :row] = 255
    row_covered = row * h * col_steps
    col = floor((r_val - row_covered) / col_steps)
    memory[:col, row] = 255
    col_covered = col * col_steps
    last_pixel = floor(r_val - row_covered - col_covered) 
    memory[col, row] = last_pixel * (255 // col_steps)
    return memory

def create_info_bars(progress_reward, w, h, col_steps):
    level = progress_reward['level'] * 100
    hp = progress_reward['hp'] * 2000 
    explore = progress_reward['explore'] * 160
    
    badges = progress_reward['badge']
    
    full_memory = np.stack((
        make_reward_channel(level, w, h, col_steps),
        make_reward_channel(hp, w, h, col_steps),
        make_reward_channel(explore, w, h, col_steps),
    ), axis=-1)
    
    # add progress bar for badges from the right
    if badges > 0:
        full_memory[:, -1, :] = 255

    return full_memory
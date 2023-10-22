from math import floor

import numpy as np

def make_reward_channel(r_val, w, h, col_steps):
    row = floor(r_val / (h * col_steps))
    memory = np.zeros(shape=(h, w), dtype=np.uint8)
    memory[:row, :] = 255
    row_covered = row * h * col_steps
    col = floor((r_val - row_covered) / col_steps)
    memory[:col, row] = 255
    col_covered = col * col_steps
    last_pixel = floor(r_val - row_covered - col_covered)
    memory[col, row] = last_pixel * (255 // col_steps)
    return memory.reshape(h, w, 1)

def create_info_bars(progress_reward, w, h, col_steps):
    bar_height = h // 3
    remainder = h % 3

    level = min(progress_reward['level'] * 100, w)
    hp = min(progress_reward['hp'] * w, w)
    explore = min(progress_reward['explore'] * 160, w)
    badges = progress_reward['badge']

    level_bar = make_reward_channel(level, w, bar_height, col_steps)
    hp_bar = make_reward_channel(hp, w, bar_height, col_steps)
    explore_bar = make_reward_channel(explore, w, bar_height, col_steps)

    # Handling the remainder by filling the last bar up to h
    if remainder > 0:
        explore_bar = np.pad(explore_bar, ((0, remainder), (0, 0), (0, 0)), 'constant', constant_values=0)

    full_memory = np.vstack((level_bar, hp_bar, explore_bar))

    # Add progress bar for badges from the right
    if badges > 0:
        full_memory[:, -1, :] = 255

    return full_memory
from math import floor

import numpy as np
from ConfigToAttr import apply_dict_as_attributes
from gymnasium import spaces
from PokeRed import PokeRed
from skimage.transform import resize


class EnvInputConstructor:
    ENV_CONFIG = {
        "frame_shaoe": (36, 40, 1),
        "mem_padding": 2,
        "mem_height": 8,
        "combined_shape": (46, 40, 1),
        "col_steps": 16,
        "metadata": {"render.modes": []},
        "reward_range": (0, 15000),
        "action_space": spaces.Discrete(len(PokeRed.VALID_ACTIONS)),
        "observation_space": spaces.Box(
            low=0, high=255, shape=(46, 40, 1), dtype=np.uint8
        ),
    }

    def __init__(self):
        apply_dict_as_attributes(self, self.ENV_CONFIG)
        self.pad = np.zeros(
            shape=(self.mem_padding, self.combined_shape[1], 1), dtype=np.uint8
        )

    def render_for_ml(self, stats, frame, last_rewards):
        formatted_rewards = {k: f"{v:3.2f}" if (isinstance(v, float) or isinstance(v, int)) else v for k, v in last_rewards.items()}
        #print(f"Last Rewards: {formatted_rewards}", flush=True)
        rendered_frame = self.scale_frame(frame)
        rendered_infobars = self.get_infobars(last_rewards)
        full_ml_image = np.concatenate(
            (rendered_infobars, self.pad, rendered_frame), axis=0
        )
        return full_ml_image

    def scale_frame(self, frame, resolution=(36, 40)):
        scaled = (255 * resize(frame, resolution, anti_aliasing=True)).astype(np.uint8)
        scaled = scaled[:, :, :1]
        return scaled

    def get_infobars(self, last_rewards):
        info_bars = self.create_info_bars(
            last_rewards, self.combined_shape[1], self.mem_height, self.col_steps
        )
        return info_bars

    def create_info_bars(self, progress_reward, w, h, col_steps):
        bar_height = h // 3
        remainder = h % 3

        level = (
            min(progress_reward["level"] * 100, w) if "level" in progress_reward else 0
        )
        hp = min(progress_reward["hp"] * w, w) if "hp" in progress_reward else 0
        explore = (
            min(progress_reward["explore"] * 160, w)
            if "explore" in progress_reward
            else 0
        )
        badges = progress_reward["badge"] if "badge" in progress_reward else 0

        level_bar = self.make_reward_channel(level, w, bar_height, col_steps)
        hp_bar = self.make_reward_channel(hp, w, bar_height, col_steps)
        explore_bar = self.make_reward_channel(explore, w, bar_height, col_steps)

        # Handling the remainder by filling the last bar up to h
        if remainder > 0:
            explore_bar = np.pad(
                explore_bar,
                ((0, remainder), (0, 0), (0, 0)),
                "constant",
                constant_values=0,
            )

        full_memory = np.vstack((level_bar, hp_bar, explore_bar))

        # Add progress bar for badges from the right
        if badges > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def make_reward_channel(self, r_val, w, h, col_steps):
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

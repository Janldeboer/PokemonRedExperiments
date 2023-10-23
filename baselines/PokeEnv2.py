from math import floor

import numpy as np

from skimage.transform import resize

class EnvInputConstructor:
    def __init__(*args):
        pass

    def transform_for_model(self, stats, frame):
        scaled_frame = self.scale_frame(frame)
        agent_stats = self.get_agent_stats(stats)
        return scaled_frame, agent_stats
    
    def scale_frame(self, frame, resolution=(36, 40)):
        scaled = (255*resize(frame, resolution, anti_aliasing=True)).astype(np.uint8)
        return scaled
            
    
    def get_agent_stats(self, stats):
        """ Retrieve and scale selected stats for the agent 
        Return a numpy array with shape (152, 6)
        """
    
        # return empty placeholder
        return np.zeros(shape=(152, 6), dtype=np.uint8)

    def render_for_ml(self, frame):
        pixels = self.render()
        pixels = self.add_infobar_to_render()
        return pixels
    
    def render_stats(self, stats):
        pass

    def add_infobar_to_render(self):
        pad = np.zeros(shape=(self.mem_padding, self.output_shape[1], 1), dtype=np.uint8)
        info_bars = self.create_info_bars(self.last_rewards, self.output_shape[1], self.memory_height, self.col_steps)
        full_image =  np.concatenate((info_bars, pad, self.last_frame), axis=0)
        return full_image
    
    def create_info_dot(self, red_val, green_val, blue_val, w, h):
        memory = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        memory[:, :, 0] = red_val
        memory[:, :, 1] = green_val
        memory[:, :, 2] = blue_val
        return memory
    
    def create_poke_dot(self, poke_index, w=3, h=3):
        pass
    

    def make_reward_channelse(self, r_val, w, h, col_steps):
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

    def create_info_bars(self, progress_reward, w, h, col_steps):
        bar_height = h // 3
        remainder = h % 3
        
        level = min(progress_reward['level'] * 100, w) if 'level' in progress_reward else 0
        hp = min(progress_reward['Relative HP'] * w, w) if 'Relative HP' in progress_reward else 0
        explore = min(progress_reward['explore'] * 160, w) if 'explore' in progress_reward else 0
        badges = progress_reward['badge'] if 'badge' in progress_reward else 0

        level_bar = self.make_reward_channel(level, w, bar_height, col_steps)
        hp_bar = self.make_reward_channel(hp, w, bar_height, col_steps)
        explore_bar = self.make_reward_channel(explore, w, bar_height, col_steps)

        # Handling the remainder by filling the last bar up to h
        if remainder > 0:
            explore_bar = np.pad(explore_bar, ((0, remainder), (0, 0), (0, 0)), 'constant', constant_values=0)

        full_memory = np.vstack((level_bar, hp_bar, explore_bar))

        # Add progress bar for badges from the right
        if badges > 0:
            full_memory[:, -1, :] = 255

        return full_memory
import json
import sys
import uuid
from math import floor
from pathlib import Path

import hnswlib
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd
from einops import rearrange
from pyboy import PyBoy
from skimage.transform import resize

from gymnasium import Env, spaces
from PokeRedReader import PokeRedReader
from PokemonRedRewarder import PokemonRedRewarder
from KnnHandler import KnnHandler
from pyboy.utils import WindowEvent

DEFAULTS_PATH = './default_config.json'

with open(DEFAULTS_PATH, 'r') as f:
    DEFAULTS = json.load(f)

class RedGymEnv(Env):

    VALID_ACTIONS = [
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
        WindowEvent.PRESS_BUTTON_START,
        WindowEvent.PASS,
    ]

    RELEASE_ACTIONS = [
        WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.RELEASE_BUTTON_B,
        WindowEvent.RELEASE_BUTTON_START,
    ]

    def __init__(
        self, config=None):
        config = {**DEFAULTS, **(config or {})}  # Merge defaults with provided config

        for key, value in config.items():
            setattr(self, key, value)

        self.video_interval = 256 * self.action_freq
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.session_path.mkdir(exist_ok=True)


        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.VALID_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.poke_reader = PokeRedReader(pyboy = self.pyboy)
        self.poke_rewarder = PokemonRedRewarder(poke_reader = self.poke_reader, save_screenshot = self.save_screenshot)
        self.knn_handler = KnnHandler(poke_reader= self.poke_reader)
        self.screen = self.pyboy.botsupport_manager().screen()

        self.pyboy.set_emulation_speed(0 if config['headless'] else 6)
        self.reset()

    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        print(f'Init state: {self.init_state}')
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.knn_handler = KnnHandler(poke_reader= self.poke_reader)

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []
        
        if self.save_video:
            base_dir = self.session_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()
       
        self.levels_satisfied = False
        self.step_count = 0
        self.progress_reward = self.poke_rewarder.get_game_state_reward()
        self.progress_reward['explore'] = self.knn_handler.get_knn_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self.render(), {}
    
    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(), 
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render
    
    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
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
        
        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self.poke_reader.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height)
        
    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
            frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)

        self.knn_handler.update_frame_knn_index(obs_flat)
            
        self.poke_rewarder.update_heal_reward()

        new_reward, new_prog = self.update_reward()
        
        self.poke_rewarder.last_health = self.poke_reader.read_hp_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.VALID_ACTIONS[action])
        for i in range(self.action_freq):

            if i == 8 and action < 7:
                self.pyboy.send_input(self.RELEASE_ACTIONS[action])


            if self.save_video and not self.fast_video:
                self.add_video_frame()
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))
        
    def finish_video(self):
        self.full_frame_writer.close()
        self.model_frame_writer.close()
    
    def append_agent_stats(self, action):
        x_pos = self.poke_reader.read_m(0xD362)
        y_pos = self.poke_reader.read_m(0xD361)
        map_n = self.poke_reader.read_m(0xD35E)
        levels = [self.poke_reader.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        self.agent_stats.append({
            'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'last_action': action,
            'pcount': self.poke_reader.read_m(0xD163), 'levels': levels, 'ptypes': self.poke_reader.read_party(),
            'hp': self.poke_reader.read_hp_fraction(),
            'frames': self.knn_handler.knn_index.get_current_count(),
            'deaths': self.poke_rewarder.died_count, 'badge': self.poke_reader.get_badges(),
            'event': self.progress_reward['event'], 'healr': self.poke_rewarder.total_healing_rew,
        })
    
    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.poke_rewarder.get_game_state_reward()
        self.progress_reward['explore'] = self.knn_handler.get_knn_reward()
        new_prog = self.group_rewards()
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.poke_reader.read_hp_fraction() > 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')
    
        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )
    
    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (prog['level'] * 100, self.poke_reader.read_hp_fraction()*2000, prog['explore'] * 160)#(prog['events'], 
               # prog['levels'] + prog['party_xp'], 
               # prog['explore'])


    def check_if_done(self):
        if self.early_stop:
            return self.step_count > 128 and self.recent_memory.sum() < (255 * 1)
        else:
            return self.step_count >= self.max_steps

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.session_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))

        if done:
            self.clean_up(obs_memory)
            
    def clean_up(self, obs_memory):
        self.all_runs.append(self.progress_reward)
        with open(self.session_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
            json.dump(self.all_runs, f)
        pd.DataFrame(self.agent_stats).to_csv(
            self.session_path / Path(f'afent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')
                
        if self.print_rewards:
            print('', flush=True)
            if self.save_final_state:
                fsession_path = self.session_path / Path('final_states')
                fsession_path.mkdir(exist_ok=True)
                plt.imsave(
                    fsession_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fsession_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False))

        if self.save_video:
            self.finish_video()
    
    def save_screenshot(self, name):
        ss_dir = self.session_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render(reduce_res=False))
    

    



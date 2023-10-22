import json
import sys
import uuid
from math import floor, prod
from pathlib import Path

import hnswlib
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd
from einops import rearrange
from skimage.transform import resize

from gymnasium import Env spaces
from PokeRedReader import PokeRedReader
from PokeRedRewarder import PokeRedRewarder
from PokeRecorder import PokeRecorder
from PokeRed import PokeRed

from stable_baselines3.common.utils import set_random_seed

from PokeReadInputLayer import create_info_bars

DEFAULTS_PATH = './default_config.json'

with open(DEFAULTS_PATH, 'r') as f:
    DEFAULTS = json.load(f)
    
def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

# Coming soon: A wrapper encapsulating the gym environment, pokemon red, and the rewarder
class PokemonRedRLSuite:
    def __init__(self):
        pass

class RedGymEnv(Env):
    def __init__(self, config=None):
        self.load_config(config)
        
        self.setup_enivronment()
        self.poke_red = PokeRed(self.gb_path, state_file=self.init_state, head=self.head)

        self.poke_rewarder = PokeRedRewarder()
        self.vec_dim = prod(self.output_shape)
        self.last_screen = None
        self.levels_satisfied = False
        self.last_frame = None
        
        if self.save_video:
            self.recorder = PokeRecorder(self.session_path, self.instance_id, self.output_shape, self.render, self.render_for_ml, self.reset_count)
    
        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(PokeRed.VALID_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        self.reset()
        
    def setup_enivronment(self):
        self.output_shape = (36, 40, 1)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + self.memory_height + self.mem_padding,
                            self.output_shape[1],
                            self.output_shape[2]
        ) 
    
        print(f'output shape: {self.output_full}')
        
    def load_config(self, config):
        config = {**DEFAULTS, **(config or {})}
        for key, value in config.items():
            setattr(self, key, value)
            
        self.video_interval = 256 * self.act_freq
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.session_path.mkdir(exist_ok=True)
        self.head = 'headless' if self.headless else 'SDL2'
            
    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        self.poke_red.load_from_state(self.init_state)
        self.poke_rewarder.reset()
        
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.finish_video()
            self.recorder = PokeRecorder(self.session_path, self.instance_id, self.output_shape, self.render, self.render_for_ml, self.reset_count)
       
        self.levels_satisfied = False
        self.step_count = 0
        self.last_rewards = {}
        self.total_reward = 1
        self.reset_count += 1
        return self.render_for_ml(), {}
    
    def render(self, reduce_res=True, update_mem=True):
        resolution = self.output_shape if reduce_res else None
        pixels = self.poke_red.get_screen(resolution)
        if reduce_res and update_mem:
            self.last_frame = pixels
        return pixels
    
    def render_for_ml(self, update_mem=True):
        pixels = self.render(reduce_res=True, update_mem=update_mem)
        pixels = self.add_infobar_to_render()
        return pixels

    def add_infobar_to_render(self):
        pad = np.zeros(shape=(self.mem_padding, self.output_shape[1], 1), dtype=np.uint8)
        info_bars = create_info_bars(self.last_rewards, self.output_shape[1], self.memory_height, self.col_steps)
        full_image =  np.concatenate((info_bars, pad, self.last_frame), axis=0)
        return full_image
    
    def step(self, action):

        self.poke_red.run_action_on_emulator(action)

        obs_memory, obs_flat = self.get_agent_state(self.poke_red.get_all_stats())
        
        new_reward = self.poke_rewarder.update_rewards(self.poke_red.get_all_stats(), obs_flat)
        #self.update_recent_memory(new_prog)
        step_limit_reached = self.check_if_done()
        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}
    
    def get_agent_state(self, game_state):
        #self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render_for_ml()

        # trim off memory from frame for knn index
        frame_start = self.memory_height + self.mem_padding
        obs_flat = obs_memory[
            frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)
            
        return obs_memory, obs_flat

    def check_if_done(self):
        return self.step_count >= self.max_steps

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.last_rewards.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.session_path / Path(f'curframe_{self.step_count}_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))

        if done:
            self.clean_up(obs_memory)
            
    def clean_up(self, obs_memory):
        self.all_runs.append(self.last_rewards)
        with open(self.session_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
            json.dump(self.all_runs, f)
            
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

        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.finish_video()
    
    def save_screenshot(self, name):
        ss_dir = self.session_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render(reduce_res=False))
    
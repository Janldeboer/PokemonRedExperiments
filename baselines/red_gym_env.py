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
from skimage.transform import resize

from gymnasium import Env, spaces
from PokeRedReader import PokeRedReader
from PokeRedRewarder import PokeRedRewarder
from PokeRecorder import PokeRecorder
from PokeRed import PokeRed
from KnnHandler import KnnHandler

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

class RedGymEnv(Env):
    def __init__(self, config=None, new_shape=False):
        self.load_config(config)
        
        self.setup_enivronment(new_shape=new_shape)
        self.poke_red = PokeRed(self.gb_path, state_file=self.init_state, head=self.head)

        self.poke_rewarder = PokeRedRewarder()
        self.knn_handler = KnnHandler()
        self.last_screen = None
        self.levels_satisfied = False
        
        if self.save_video:
            self.recorder = PokeRecorder(self.session_path, self.instance_id, self.output_shape, self.render, self.render_for_ml, self.reset_count)
    
        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(PokeRed.VALID_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        self.reset()
        
    def setup_enivronment(self, new_shape):
        if new_shape:
            self.output_shape = (36, 40, 3)
            self.mem_padding = 2
            self.memory_height = 8
            self.col_steps = 16
            self.output_full = (
                self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                                self.output_shape[1],
                                self.output_shape[2]
            )
        else:
            # This defines the whole observation space for the agent
            # Changes is input and encoding should be reflected here
            self.output_shape = (36, 40, 3)
            self.mem_padding = 2
            self.memory_height = 8
            self.col_steps = 16
            self.output_full = (
                self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                                self.output_shape[1],
                                self.output_shape[2]
            ) # (128, 40, 3) : 36 * 3  + 2 * (2 + 8)
             # 128 * 40 * 3 = 15_360
        # n = 10
        # self.feature_vector_length = n  # replace n with the length of your feature vector
        # self.output_full = (
        #     self.output_shape[0],
        #     self.output_shape[1] + self.feature_vector_length,
        #     self.output_shape[2]
        # )
        
        print(f'output shape: {self.output_full}')
        
    def load_config(self, config):
        config = {**DEFAULTS, **(config or {})}
        for key, value in config.items():
            setattr(self, key, value)
            
        self.video_interval = 256 * self.act_freq
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.session_path.mkdir(exist_ok=True)
        self.head = 'headless' if self.headless else 'SDL2'
            
    def initialize_state(self):
        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
            self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)
            
    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        self.poke_red.load_from_state(self.init_state)

        self.knn_handler = KnnHandler()
        if not self.use_screen_explore:
            self.init_map_mem()

        self.initialize_state()

        self.agent_stats = []
        
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.finish_video()
       
        self.levels_satisfied = False
        self.step_count = 0
        self.progress_reward = self.poke_rewarder.get_rewards()
        foobar = None if self.use_screen_explore else len(self.seen_coords)
        self.progress_reward['explore'] = self.knn_handler.get_knn_reward(foobar)
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self.render_for_ml(), {}
    
    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True, update_mem=True):
        resolution = self.output_shape if reduce_res else None
        pixels = self.poke_red.get_screen(resolution)
        if reduce_res and update_mem:
            self.recent_frames[0] = pixels
        return pixels
    
    def render_for_ml(self, update_mem=True):
        pixels = self.render(reduce_res=True, update_mem=update_mem)
        pixels = self.add_memory_to_render()
        return pixels

    def add_memory_to_render(self):
        pad = np.zeros(shape=(self.mem_padding, self.output_shape[1], 3), dtype=np.uint8)
        info_bars = create_info_bars(self.progress_reward, self.output_shape[1], self.memory_height, self.col_steps)
        recent_memory = rearrange(self.recent_memory, '(w h) c -> h w c', h=self.memory_height)
        rearranged_frames = rearrange(self.recent_frames, 'f h w c -> (f h) w c')
        full_image =  np.concatenate((info_bars, pad, recent_memory, pad, rearranged_frames), axis=0)
        return full_image
    
    def plot_memory(self, memory):
        memory_normalized = memory.astype(float) / 255
        plt.imshow(memory_normalized)
        plt.show()
        
    def step(self, action):

        self.poke_red.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render_for_ml()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
            frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)

        if self.use_screen_explore:
            
            self.knn_handler.update_frame_knn_index(obs_flat)
            self.knn_handler.update_levels_satisfied(sum(self.poke_red.get_poke_info('Level')))
        else:
            self.update_seen_coords()
            
        new_reward, new_prog = self.update_reward()
        self.update_recent_memory(new_prog)
        step_limit_reached = self.check_if_done()
        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}
    
    def update_recent_memory(self, new_prog):
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64 * 100, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64 * 2000, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128 * 160, 255)
    
    def append_agent_stats(self, action):
        agent_stats = self.poke_red.get_agent_stats()
        agent_stats['last_action'] = action
        agent_stats['step'] = self.step_count
      
        if self.use_screen_explore:
            expl = ('frames', self.knn_handler.knn_index.get_current_count())
        else:
            expl = ('coord_count', len(self.seen_coords))
            
        agent_stats[expl[0]] = expl[1]
        agent_stats['deaths'] = self.poke_rewarder.died_count
        agent_stats['event'] = self.progress_reward['event']
        agent_stats['healr'] = self.poke_rewarder.total_healing_rew
        
        self.agent_stats.append(agent_stats)
    
    def update_seen_coords(self):
        x_pos = self.poke_red.get_stat('X')
        y_pos = self.poke_red.get_stat('Y')
        map_n = self.poke_red.get_stat('Map')
        
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if sum(self.poke_red.get_poke_info('Level')) >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}
        
        self.seen_coords[coord_string] = self.step_count

    def update_reward(self):
        level = self.progress_reward['level']
        hp = self.poke_red.read_hp_fraction()
        explore = self.progress_reward['explore']
        
        self.progress_reward = self.poke_rewarder.update_rewards({"hp": hp})
        foobar = None if self.use_screen_explore else len(self.seen_coords)
        self.progress_reward['explore'] = self.knn_handler.get_knn_reward(foobar)
        # compute reward
        old_prog = level, hp, explore

        new_prog = level, hp, explore
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.poke_red.read_hp_fraction() > 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')
    
        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )

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

        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.finish_video()

                
    def get_all_events_reward(self): # TODO: Check what of this to merge, function was moved
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )

    
    def save_screenshot(self, name):
        ss_dir = self.session_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render(reduce_res=False))
    
import json
import uuid
from gymnasium import Env
from PokeRedRewarder import PokeRedRewarder
from PokeRed import PokeRed

from EnvInputConstructor import EnvInputConstructor
from stable_baselines3.common.utils import set_random_seed

from ConfigToAttr import apply_dict_as_attributes

DEFAULTS_PATH = './default_config.json'

with open(DEFAULTS_PATH, 'r') as f:
    DEFAULTS = json.load(f)

class RedGymEnv(Env):
    def __init__(self, config=None):
        self.load_config(config)
        
        self.poke_red = PokeRed(self.gb_path, state_file=self.init_state, head=self.head)
        self.poke_rewarder = PokeRedRewarder()
        self.env_input_constructor = EnvInputConstructor()
        
        apply_dict_as_attributes(self, EnvInputConstructor.ENV_CONFIG)
        
        self.reset()

    def load_config(self, config):
        config = {**DEFAULTS, **(config or {})}
        apply_dict_as_attributes(self, config)
            
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.session_path.mkdir(exist_ok=True)
        self.head = 'headless' if self.headless else 'SDL2'
            
    def reset(self, seed=None):
        self.seed = seed
        # (re)start game, skipping credits
        self.poke_red.load_from_state(self.init_state)
        self.poke_rewarder.reset()
        self.step_count = 0
        
        # getting first observation
        stats, frame = self.poke_red.get_all_stats(), self.poke_red.get_screen()
        rewards = self.poke_rewarder.update_rewards(stats, frame)
        observation = self.env_input_constructor.render_for_ml(stats, frame, rewards)
        
        self.reset_count += 1
        return observation, {}
    
    def step(self, action):
        stats, frame = self.poke_red.run_action_on_emulator(action)
        rewards = self.poke_rewarder.update_rewards(stats, frame)
        observation = self.env_input_constructor.render_for_ml(stats, frame, rewards)
        
        step_limit_reached = self.increase_step_count()
        
        return observation, rewards['total']*0.1, False, step_limit_reached, {}
    
    def increase_step_count(self):
        """ Increase the step count by 1  and returns if the step limit has been reached."""
        self.step_count += 1
        return self.check_if_done()

    def check_if_done(self):
        return self.step_count >= self.max_steps
    
    def render(self, **kwargs):
        # check if there are any kwargs, just curious
        if len(kwargs) > 0:
            print(f"render kwargs: {kwargs}")
            
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

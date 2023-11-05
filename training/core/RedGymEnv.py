import json
import uuid

from ConfigToAttr import apply_dict_as_attributes
from EnvInputConstructor import EnvInputConstructor
from gymnasium import Env
from pathlib import Path
from PokeRed import PokeRed
from PokeRedRewarder import PokeRedRewarder
from stable_baselines3.common.utils import set_random_seed
from future.PokeRecorder import ScreenshotRecorder
from datetime import datetime

DEFAULTS_PATH = "./default_config.json"

with open(DEFAULTS_PATH, "r") as f:
    DEFAULTS = json.load(f)


class RedGymEnv(Env):
    def __init__(self, config=None):
        self.load_config(config)

        self.poke_red = PokeRed(
            self.gb_path, state_file=self.init_state, head=self.head
        )
        self.poke_rewarder = PokeRedRewarder()
        self.env_input_constructor = EnvInputConstructor()
        self.game_recorder = ScreenshotRecorder(self.session_path / Path("game"), skip=255)
        self.ml_recorder = ScreenshotRecorder(self.session_path / Path("ml"), skip=255)

        apply_dict_as_attributes(self, EnvInputConstructor.ENV_CONFIG)

        self.reset()

    def load_config(self, config):
        config = {**DEFAULTS, **(config or {})}
        apply_dict_as_attributes(self, config)

        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.session_path.mkdir(exist_ok=True)
        self.head = "headless" if self.headless else "SDL2"
    

    def reset(self, seed=None):
        self.seed = seed if seed else datetime.now().microsecond
        
        # (re)start game, skipping credits
        self.poke_red.load_from_state(self.init_state)
        self.poke_rewarder.reset()
        self.last_total_reward = 0
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
        
        reward_for_step = rewards["total"] - self.last_total_reward
        self.last_total_reward = rewards["total"]

        step_limit_reached = self.increase_step_count()
        
        image_note = f"cpu{self.rank}_s{self.step_count}_r{rewards['total']:4f}_a{action}"
        self.game_recorder.add(frame, note=image_note)
        self.ml_recorder.add(observation, note=image_note)
        
        if reward_for_step < 0 or reward_for_step > 1:
            print(f"Reward for step: {reward_for_step:4f}")
            
        """
        if reward_for_step < 0 and self.last_total_reward < 20:
            # No negative rewards in the beginning, no penalty for loosing health or dying
            # We dont want AI to avoid that and be scared of fighting
            reward_for_step = 0
        """
            
        return observation, reward_for_step, False, step_limit_reached, {}

    def increase_step_count(self):
        """Increase the step count by 1  and returns if the step limit has been reached."""
        self.step_count += 1
        return self.check_if_done()

    def check_if_done(self):
        return self.step_count >= self.max_steps

    def render(self, **kwargs):
        # check if there are any kwargs, just curious
        if len(kwargs) > 0:
            print(f"render kwargs: {kwargs}")
        else:
            print("render kwargs: None")
        return self.poke_red.get_screen()


def make_env(rank, env_conf, seed=0):
    seed = datetime.now().microsecond if seed == 0 else seed
    def _init():
        env_conf["rank"] = rank
        env_conf["seed"] = seed
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init

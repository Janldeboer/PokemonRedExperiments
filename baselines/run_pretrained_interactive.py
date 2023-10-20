from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from argparse_pokemon import *

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
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    sess_path = f'session_{str(uuid.uuid4())[:8]}'
    ep_length = 2**16
    args = get_args(usage_string=None, headless=False, ep_length=ep_length, sess_path=sess_path)

    env_config = {
        'headless': False,
        'early_stop': False,
        'max_steps': ep_length, 
        'save_video': False,
        'fast_video': True, 
        'session_path': sess_path,
    }
    env_config = change_env(env_config, args)
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    #env_checker.check_env(env)
    file_name = 'baselines/session_b30478f4/poke_49741824_past_gym1'
    print(f"file_name: {file_name}.zip")
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999)
    
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        action = 7 # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

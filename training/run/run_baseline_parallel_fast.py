from os.path import exists
from pathlib import Path
import uuid
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append('../core')
from RedGymEnv import RedGymEnv, make_env

def main():
    ep_length = 2048 * 8
    session_id = str(uuid.uuid4())
    sess_path = Path(f'session_{session_id[:8]}')
    
    print(f"Session id: {session_id[:8]}")

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../../states/has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': False, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3, # 2.5,
                'frame_stacks': 1,
            }
    
    print(env_config)
    
    num_cpu = 4  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    #env_checker.check_env(env)
    learn_steps = 10
    # put a checkpoint here you want to start from
    file_name = 'nope' #demo_session/poke_439746560_steps' 
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('\ncreating new model')
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=69, callback=checkpoint_callback)

if __name__ == '__main__':
    main()
    
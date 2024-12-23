import sys
import uuid
from datetime import datetime
from os.path import exists
from pathlib import Path

from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.append("../core")
from RedGymEnv import RedGymEnv, make_env


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    sess_path = f"../sessions/new_session_{get_timestamp()}__{str(uuid.uuid4())[:8]}"
    sess_path = Path(sess_path)
    ep_length = 2**16

    env_config = {
        "gb_path": "../../PokemonRed.gb",
        "init_state": "../../states/has_pokedex_nballs.state",
        "headless": False,
        "early_stop": False,
        "max_steps": ep_length,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "use_screen_explore": True,
        "extra_buttons": True,
        "frame_stacks": 1,
    }

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    env = make_env(
        0, env_config
    )()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # env_checker.check_env(env)
    file_name = "2nd_Run"  # 'baselines/session_b30478f4/poke_49741824_past_gym1'
    print(f"file_name: {file_name}.zip")
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(
            file_name, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0}
        )
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=ep_length,
            batch_size=512,
            n_epochs=1,
            gamma=0.999,
        )

    # keyboard.on_press_key("M", toggle_agent)
    obs = env.reset()
    while True:
        action = 7  # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except BaseException:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, info = env.step(action)
        env.render()
        if terminated:
            break
    env.close()

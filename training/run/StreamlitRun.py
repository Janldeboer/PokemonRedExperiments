import streamlit as st
import uuid
from pathlib import Path
from datetime import datetime
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append("../core")
from RedGymEnv import make_env  # Adjust this import based on your setup

if 'run_on' not in st.session_state:
    st.session_state.run_on = False

def init_streamlit():
    st.sidebar.title("Configuration")
    exponent = st.sidebar.slider('Episode Length (2^x)', min_value=0, max_value=20, value=12, step=1)
    max_steps = 2 ** exponent
    st.sidebar.text(f"Actual Episode Length: {max_steps}")
    init_state = st.sidebar.text_input('Initial State', value='../../states/has_pokedex_nballs.state')
    gb_path = st.sidebar.text_input('GameBoy Path', value='../../PokemonRed.gb')
    model_path = st.sidebar.text_input('Model Path', value='../../first_full_run.zip')
    run_on = st.sidebar.checkbox('Run Model', value=False)
    return max_steps, init_state, gb_path, model_path, run_on 

def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def setup_environment_and_model(ep_length, init_state, gb_path, model_path=None):
    session_id = str(uuid.uuid4())
    sess_path = Path(f"sessions/session_{get_timestamp()}_{session_id[:8]}")
    
    env_config = {
        "headless": True, "save_final_state": True, "early_stop": False, "action_freq": 24,
        "max_steps": ep_length, "print_rewards": True, "save_video": False, "fast_video": True,
        "session_path": sess_path, "debug": False, "sim_frame_dist": 2_000_000.0, "use_screen_explore": False,
        "reward_scale": 4, "extra_buttons": False, "explore_weight": 3, "frame_stacks": 1, "init_state": init_state, "gb_path": gb_path
    }
    
    num_cpu = 1
    if num_cpu > 1:
        env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    else:
        env = make_env(0, env_config)()
    
    if (Path(model_path)).exists():
        model = PPO.load(model_path, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # show error message
        st.error(f"Model path {model_path} does not exist")
    
    return model, env

def toggle_run(ep_length, init_state, gb_path, model_path):
    if st.session_state.run_on:
        st.session_state.run_on = False
    else:
        st.session_state.run_on = True
        run_model(ep_length, init_state, gb_path, model_path)

def run_model(ep_length, init_state, gb_path, model_path):
    if st.session_state.run_on:
        model, env = setup_environment_and_model(ep_length, init_state, gb_path, model_path)
        obs = model.env.reset()
        image_placeholder = st.empty()  # Create the placeholder once here
        while st.session_state.run_on:
            action = 7  # default action
            try:
                with open("agent_enabled.txt", "r") as f:
                    agent_enabled = f.readlines()[0].startswith("yes")
            except:
                agent_enabled = False
            if agent_enabled:
                action, _ = model.predict(obs, deterministic=False)
            obs, a, terminated, b = model.env.step(action)
            print(f"Action: {action}, Reward: {a}, Terminated: {terminated}, Info: {b}")
            frame = env.render(mode="rgb_array")
            print(f"Shape: {frame.shape}")

            # Update the placeholder with the new frame
            image_placeholder.image(frame, channels="RGB", use_column_width=True)

            model.env.render()
            if terminated.any():
                break
        model.env.close()

if __name__ == "__main__":
    ep_length, init_state, gb_path, model_path, run_on = init_streamlit()
    if run_on:
        run_model(ep_length, init_state, gb_path, model_path)



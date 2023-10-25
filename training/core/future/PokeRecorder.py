import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np

class Recorder:
    def __init__(self, path, format='gif', fps=1): # Format can be gif, png, mp4
        self.path = path
        self.format = format
        self.fps = fps
        if not format in ['gif', 'png', 'mp4']:
            raise ValueError(f'Format {format} not supported')
        
        self.setup()
    
    def setup(self):
        if self.format == 'gif':
            self.writer = media.GIFWriter(self.path, fps=self.fps)
        elif self.format == 'png':
            self.writer = media.ImageWriter(self.path)
        elif self.format == 'mp4':
            self.writer = media.VideoWriter(self.path, fps=self.fps)
        self.writer.__enter__()
        
class ScreenshotRecorder():
    def __init__(self, path, skip=0):
        self.path = path
        if not type(self.path) == Path:
            self.path = Path(self.path)
        self.path.mkdir(exist_ok=True)
        self.count = 0
        self.skip = skip
        self.skipped = 0
        
    def add(self, frame, note=None):
        self.count += 1
        if self.skipped < self.skip:
            self.skipped += 1
            return
        self.skipped = 0
        
        note = "_" + note if note else ''
        filename = f"frame{self.count}{note}.png"

        
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        
        plt.imsave(self.path / Path(filename), frame) 
            
class PokeRecorder:
    def __init__(
        self, session_path, instance_id, ml_res, renderFull, renderModel, reset_count=0
    ):
        base_dir = session_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"{base_dir}/full_reset_{reset_count}_id{instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"{base_dir}/model_reset_{reset_count}_id{instance_id}"
        ).with_suffix(".mp4")
        self.full_recorder = VideoRecorder(full_name, (144, 160), renderFull)
        self.model_recorder = VideoRecorder(model_name, ml_res[:2], renderModel)
        self.all_runs = []

    def add_video_frame(self):
        self.full_recorder.add_video_frame()
        self.model_recorder.add_video_frame()

    def finish_video(self):
        self.full_recorder.finish_video()
        self.model_recorder.finish_video()

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.last_rewards.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.session_path
                / Path(f"curframe_{self.step_count}_{self.instance_id}.jpeg"),
                self.render(reduce_res=False),
            )

        if done:
            self.clean_up(obs_memory)

    def clean_up(self, obs_memory):
        self.all_runs.append(self.last_rewards)
        with open(
            self.session_path / Path(f"all_runs_{self.instance_id}.json"), "w"
        ) as f:
            json.dump(self.all_runs, f)

        if self.print_rewards:
            print("", flush=True)
            if self.save_final_state:
                fsession_path = self.session_path / Path("final_states")
                fsession_path.mkdir(exist_ok=True)
                plt.imsave(
                    fsession_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_small.png"
                    ),
                    obs_memory,
                )
                plt.imsave(
                    fsession_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.png"
                    ),
                    self.render(reduce_res=False),
                )

        if hasattr(self, "recorder") and self.recorder:
            self.recorder.finish_video()

    def save_screenshot(self, name):
        ss_dir = self.session_path / Path("screenshots")
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir
            / Path(
                f"frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg"
            ),
            self.render(reduce_res=False),
        )


class VideoRecorder:
    def __init__(self, path, resolution, render, fps=60):
        self.writer = media.VideoWriter(path, resolution, fps=fps)
        self.writer.__enter__()
        self.render = render

    def add_video_frame(self):
        self.writer.add_image(self.render)

    def finish_video(self):
        self.writer.close()

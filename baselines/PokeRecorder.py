
from pathlib import Path
import mediapy as media


class PokeRecorder:
    
    def __init__(self, session_path, instance_id, ml_res, renderFull, renderModel, reset_count=0):
        base_dir = session_path / Path('rollouts')
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f'{base_dir}/full_reset_{reset_count}_id{instance_id}').with_suffix('.mp4')
        model_name = Path(f'{base_dir}/model_reset_{reset_count}_id{instance_id}').with_suffix('.mp4')
        self.full_recorder = VideoRecorder(full_name, (144, 160), renderFull)
        self.model_recorder = VideoRecorder(model_name, ml_res[:2], renderModel)
        
    def add_video_frame(self):
        self.full_recorder.add_video_frame()
        self.model_recorder.add_video_frame()
        
    def finish_video(self):
        self.full_recorder.finish_video()
        self.model_recorder.finish_video()
        
class VideoRecorder:
    def __init__(self, path, resolution, render, fps=60):
        self.writer = media.VideoWriter(path, resolution, fps=fps)
        self.writer.__enter__()
        self.render = render
        
    def add_video_frame(self):
        self.writer.add_image(self.render)
        
    def finish_video(self):
        self.writer.close()
    
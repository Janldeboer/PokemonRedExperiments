
import hnswlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

class KnnHandler:
    def __init__(self, num_elements=20000, vec_dim=4320):
        self.num_elements = num_elements
        self.vec_dim = vec_dim
        
        self.base_explore = 0
        self.levels_satisfied = False
        self.sim_frame_dist = 2000000
        
        self.knn_index = self.create_index()
        
    def create_index(self):
        new_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        new_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)
        return new_index
        
    # This seems to split exploration in a base and post phase
    # I'm not sure why, but will keep it for now
    def update_levels_satisfied(self, levels_sum):
        if levels_sum >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.knn_index = self.create_index()
            
    def is_frame_novel(self, vec):
        if self.knn_index.get_current_count() == 0:
            print("adding first frame to knn index")
            return True
        else:
            return self.knn_index.knn_query(vec, k = 1)[1][0] > self.sim_frame_dist
        
    def update_frame_knn_index(self, frame_vec):
        if self.is_frame_novel(frame_vec):
            new_frame_id = self.knn_index.get_current_count()
            identifiers = np.array([new_frame_id])
            self.knn_index.add_items(frame_vec, identifiers)
                
    def get_knn_reward(self):
        pre_rew = 0.004
        post_rew = 0.01
        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post
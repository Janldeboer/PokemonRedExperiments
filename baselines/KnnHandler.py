
import hnswlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

class KnnHandler:
    def __init__(self, poke_reader, num_elements=1000000, vec_dim=4320, print_rewards=False, save_video=False, save_final_state=False):
        
        self.num_elements = num_elements
        self.vec_dim = vec_dim
        
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)
        
        self.poke_reader = poke_reader
        
        self.levels_satisfied = False
        self.base_explore = 0
        self.sim_frame_dist = 2000000

    def update_frame_knn_index(self, frame_vec):
        
        if self.poke_reader.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            
            # in the original code, the knn handler was completely reset here
            # i still don't know why
            self.knn_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)
            
        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            print("adding first frame to knn index")
            
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0] > self.sim_frame_dist:
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )
                
    def get_knn_reward(self):
        pre_rew = 0.004
        post_rew = 0.01
        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post
from math import prod
import numpy as np

from KnnHandler import KnnHandler
from skimage.transform import resize


class PokeRedRewarder:
    def __init__(self):
        self.max_level_rew = 0
        self.max_xp_rew = 0
        self.died_count = 0
        self.hp_fraction = 1
        self.badge = 0
        self.knn_reward = 0
        self.total_reward = 1
        self.knn_handler = None

    def reset(self):
        self.knn_handler = None

    def get_rewards(self):
        rewards = {
            "level": 2*self.max_level_rew - 12,
            "dead": -0.1 * self.died_count,
            "xp": self.max_xp_rew * 0.0_000_001,
            "Relative HP": min(1, 2 *self.hp_fraction),
            "badge": 20*self.badge,
            "explore": 0.02 * self.knn_reward,
        }
        rewards["total"] = sum([val for _, val in rewards.items()])
        return rewards

    def add_to_knn(self, frame_vec):
        scaled = (255 * resize(frame_vec, (36,40), anti_aliasing=True)).astype(np.uint8)
        scaled = scaled[:, :, :1]
        if not self.knn_handler:
            print("Creating new knn handler")
            self.knn_handler = KnnHandler(vec_dim=prod(scaled.shape))
        self.knn_handler.update_frame_knn_index(scaled)

    def update_rewards(self, new_stats, new_frame):
        
    
        self.max_level_rew = max(self.max_level_rew, sum(new_stats["Level"]))
        self.max_xp_rew = max(self.max_xp_rew, sum(new_stats["XP"]))
        self.badge = new_stats["Badges"]
        self.hp_fraction = new_stats["Relative HP"]
        self.add_to_knn(new_frame)
        self.knn_reward = self.knn_handler.count if self.knn_handler else 0

        self.total_reward = sum([val for _, val in self.get_rewards().items()])

        return self.get_rewards()

    # def update_total_reward(self, hp_fraction):
    #     self.update_rewards({"hp": hp_fraction})
    #     # compute reward
    #     old_prog = self.progress_reward['level'], hp_fraction, self.progress_reward['explore']
    #     self.progress_reward = self.poke_rewarder.get_rewards()
    #     self.progress_reward['explore'] = self.knn_handler.get_knn_reward()
    #     new_prog = self.progress_reward['level'], hp_fraction, self.progress_reward['explore']
    #     new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
    #     new_step = new_total - self.total_reward
    #     if new_step < 0 and self.poke_red.read_hp_fraction() > 0:
    #         #print(f'\n\nreward went down! {self.progress_reward}\n\n')
    #         self.save_screenshot('neg_reward')

    #     self.total_reward = new_total
    #     return (new_step,
    #                (new_prog[0]-old_prog[0],
    #                 new_prog[1]-old_prog[1],
    #                 new_prog[2]-old_prog[2])
    #            )

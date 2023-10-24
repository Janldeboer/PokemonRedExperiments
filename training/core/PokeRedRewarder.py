from KnnHandler import KnnHandler
from math import prod


class PokeRedRewarder:
    def __init__(self):
        self.max_level_rew = 0
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
            "level": self.max_level_rew,
            "dead": -0.1 * self.died_count,
            "Relative HP": self.hp_fraction,
            "badge": self.badge,
            "explore": self.knn_reward,
        }
        rewards["total"] = sum([val for _, val in rewards.items()])
        return rewards

    def add_to_knn(self, frame_vec):
        if not self.knn_handler:
            self.knn_handler = KnnHandler(vec_dim=prod(frame_vec.shape))
        self.knn_handler.update_frame_knn_index(frame_vec)

    def update_rewards(self, new_stats, new_frame):
        self.max_level_rew = max(self.max_level_rew, sum(new_stats["Level"]))
        self.badge = new_stats["Badges"]
        self.hp_fraction = new_stats["Relative HP"]
        self.knn_reward = self.knn_handler.count if self.knn_handler else 0

        self.add_to_knn(new_frame)

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

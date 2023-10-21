
class PokeRedRewarder:
    
    def __init__(self):
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.total_healing_rew = 0
        self.died_count = 0
        self.last_health = 1
        self.badge = 0
        self.total_reward = 0
        
    def get_rewards(self):
        rewards = {
            'event': self.max_event_rew,  
            'level': self.max_level_rew, 
            'heal': self.total_healing_rew,
            'op_lvl': self.max_opponent_level,
            'dead': -0.1*self.died_count,
            'hp': self.last_health,
            'badge': self.badge,
        }
        return rewards

    def update_rewards(self, new_stats):
        if 'op_lvl' in new_stats:
            self.max_opponent_level = max(self.max_opponent_level, new_stats['op_lvl'])
        if 'event' in new_stats:
            self.max_event_rew = max(self.max_event_rew, new_stats['event'])
        if 'level' in new_stats:
            self.max_level_rew = max(self.max_level_rew, new_stats['level'])
        if 'heal' in new_stats:
            self.total_healing_rew += new_stats['heal']
        if 'dead' in new_stats:
            self.died_count += new_stats['dead']
        if 'badge' in new_stats:
            self.badge = new_stats['badge']
        if 'hp' in new_stats:
            self.update_heal_reward(new_stats['hp'])
        
        return self.get_rewards()
    
    def update_heal_reward(self, cur_health):
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1
        self.last_health = cur_health

    def update_total_reward(self, hp_fraction):
        self.update_rewards({"hp": hp_fraction})
        # compute reward
        old_prog = self.progress_reward['level'], hp_fraction, self.progress_reward['explore']
        self.progress_reward = self.poke_rewarder.get_rewards()
        self.progress_reward['explore'] = self.knn_handler.get_knn_reward()
        new_prog = self.progress_reward['level'], hp_fraction, self.progress_reward['explore']
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.poke_red.read_hp_fraction() > 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')
    
        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )
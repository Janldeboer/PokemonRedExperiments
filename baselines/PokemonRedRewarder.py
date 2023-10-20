
import hnswlib

class PokemonRedRewarder:
    
    def __init__(self, poke_reader, save_screenshot):
        self.poke_reader = poke_reader
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.total_healing_rew = 0
        self.died_count = 0
        self.last_health = 1
        self.save_screenshot = save_screenshot
        
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.poke_reader.read_m(0xD163)
        poke_xps = [self.poke_reader.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.poke_reader.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.poke_reader.bit_count(self.poke_reader.read_m(i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.poke_reader.bit_count(self.poke_reader.read_m(i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.poke_reader.read_bit(0xD74E, 1) 
        oak_pokedex = self.poke_reader.read_bit(0xD74B, 5)
        opponent_level = self.poke_reader.read_m(0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.poke_reader.read_m(0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''
        
        state_scores = {
            'event': self.update_max_event_rew(),  
            #'party_xp': 0.1*sum(poke_xps),
            'level': self.get_levels_reward(), 
            'heal': self.total_healing_rew,
            'op_lvl': self.update_max_op_level(),
            'dead': -0.1*self.died_count,
            'badge': self.poke_reader.get_badges() * 2,
            #'op_poke': self.max_opponent_poke * 800,
            #'money': money * 3,
            #'seen_poke': seen_poke_count * 400,
            #'explore': self.get_knn_reward()
        }
        
        return state_scores

    def update_heal_reward(self):
        cur_health = self.poke_reader.read_hp_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1
                
    def update_max_op_level(self):
        #opponent_level = self.poke_reader.read_m(0xCFE8) - 5 # base level
        opponent_level = max(self.poke_reader.read_opponent_levels()) - 5
        #if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2
    
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew
    
    def get_all_events_reward(self):
        return max(sum([self.poke_reader.bit_count(self.poke_reader.read_m(i)) for i in range(0xD747, 0xD886)]) - 13, 0)
  
    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.poke_reader.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

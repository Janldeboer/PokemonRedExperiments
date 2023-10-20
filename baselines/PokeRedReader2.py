import argparse
import json
from pyboy import PyBoy

class PokeRedReader:

    SPRITE_DATA = {
        "PictureID": 0xD158,2
    }

    POKEMON_STATS = { # this is a new try to store this data more affectively and avoiding the need to write the same code over and over again
        "Pokemon":  { "address": 0xD16B, "length": 1 },
        "HP":       { "address": 0xD16C, "length": 2 },
        "Status":   { "address": 0xD16F, "length": 1 },
        "Type":   { "address": 0xD170, "length": 1, "amount": 2 },
        "Move":   { "address": 0xD173, "length": 1, "amount": 4 },
        "XP":   { "address": 0xD179, "length": 4 },
        "HP EV":   { "address": 0xD17C, "length": 2 },
        "Attack EV":   { "address": 0xD17E, "length": 2 },
        "Defense EV":   { "address": 0xD180, "length": 2 },
        "Speed EV":   { "address": 0xD182, "length": 2 },
        "Special EV":   { "address": 0xD184, "length": 2 },
        "Attack/Defense IV":   { "address": 0xD186, "length": 1 },
        "Speed/Special IV":   { "address": 0xD187, "length": 1 },
        "PP":   { "address": 0xD188, "length": 1, "amount": 4 },
        "Level":   { "address": 0xD18C, "length": 1 },
        "Max HP":   { "address": 0xD18D, "length": 2 },
        "Attack":   { "address": 0xD18F, "length": 2 },
        "Defense":   { "address": 0xD191, "length": 2 },
        "Speed":   { "address": 0xD193, "length": 2 },
        "Special":   { "address": 0xD195, "length": 2 },
        "Nickname":   { "address": 0xD2B5, "length": 10 }
    }

    POKEMON_OFFSET = 0x2C
    OPPONENT_STARTS = [0xD8A4]

    def __init__(self, pyboy):
        self.pyboy = pyboy

    def get_pokemion_info(self, info, pokemon_index, info_index=0):
        if not info in self.POKEMON_STATS:
            print(f"Error: {info} is not a valid pokemon info")
            return None

        pokemon_offset = self.POKEMON_OFFSET * pokemon_index

        stat_address = self.POKEMON_STATS[info]["address"] + pokemon_offset
        stat_length = self.POKEMON_STATS[info]["length"]
        stat_amount = -1 if not "amount" in self.POKEMON_STATS[info] else self.POKEMON_STATS[info]["amount"]
        if info_index >= 0 and info_index < stat_amount:
            stat_address += info_index * stat_length
        
        return self.read_multi_byte(stat_address, stat_length)    
        
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_multi_byte(self, start_addr, length):
        return sum(self.read_m(start_addr + i) * 256 ** i for i in range(length))

    def get_all_stats_for_pokemon(self, pokemon_index):
        stats = {}
        for stat in self.POKEMON_STATS:
            stats[stat] = self.get_pokemion_info(stat, pokemon_index)
        return stats

    def get_all_stats_for_all_pokemon(self):
        stats = []
        for i in range(6):
            stats.append(self.get_all_stats_for_pokemon(i))
        return stats


def parse_arguments():
    parser = argparse.ArgumentParser(description="Read Pokemon Red save state stats.")
    parser.add_argument("game_file_path", type=str, help="Path to the game ROM.")
    parser.add_argument("state_file_path", type=str, help="Path to the save state.")
    return parser.parse_args()

def scan_saved_state(game_file_path, state_file_path):
    pyboy = PyBoy(game_file_path, window_type="headless")
    with open(state_file_path, "rb") as f:
        pyboy.load_state(f)
    reader = PokeRedReader(pyboy)

    print(json.dumps(reader.get_all_stats_for_all_pokemon(), indent=4))

if __name__ == '__main__':
    args = parse_arguments()
    scan_saved_state(args.game_file_path, args.state_file_path)

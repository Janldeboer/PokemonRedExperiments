import argparse
from pyboy import PyBoy

__all__ = ['PokeRedReader']

class PokeRedReader:

    # Check this for all the addresses: 
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map

    NAME_ADDRESS = 0xD158
    MONEY_ADDRESS = 0xD347 # 3 bytes, bcd (binary coded decimal)
    # Pokemon Stats
    PARTY_ADDRESSES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
    POKEMON_ADDRESSES = [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247] # Duplicate of PARTY_ADDRESSES ?
    HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
    # NOT_LEVEL_ADDRESSES = [0xD16E, 0xD19A, 0xD1C6, 0xD1F2, 0xD21E, 0xD24A] # Not level
    STATUS_ADDRESSES = [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]
    TYPE_1_ADDRESSES = [0xD170, 0xD19C, 0xD1C8, 0xD1F4, 0xD220, 0xD24C]
    TYPE_2_ADDRESSES = [0xD171, 0xD19D, 0xD1C9, 0xD1F5, 0xD221, 0xD24D]
    MOVE_1_ADDRESSES = [0xD173, 0xD19F, 0xD1CB, 0xD1F7, 0xD223, 0xD24F] # Moves 2, 3, 4, are + 1, 2, 3
    XP_ADDRESSES = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255] # 4 bytes
    HP_EV_ADDRESSES = [0xD17C, 0xD1A8, 0xD1D4, 0xD200, 0xD22C, 0xD258] # 2 bytes
    ATTACK_EV_ADDRESSES = [0xD17E, 0xD1AA, 0xD1D6, 0xD202, 0xD22E, 0xD25A] # 2 bytes
    DEFENSE_EV_ADDRESSES = [0xD180, 0xD1AC, 0xD1D8, 0xD204, 0xD230, 0xD25C] # 2 bytes
    SPEED_EV_ADDRESSES = [0xD182, 0xD1AE, 0xD1DA, 0xD206, 0xD232, 0xD25E] # 2 bytes
    SPECIAL_EV_ADDRESSES = [0xD184, 0xD1B0, 0xD1DC, 0xD208, 0xD234, 0xD260] # 2 bytes
    ATTACK_DEFENSE_IV_ADDRESSES = [0xD186, 0xD1B2, 0xD1DE, 0xD20A, 0xD236, 0xD262]
    SPEED_SPECIAL_IV_ADDRESSES = [0xD187, 0xD1B3, 0xD1DF, 0xD20B, 0xD237, 0xD263]
    PP_1_ADDRESSES = [0xD188, 0xD1B4, 0xD1E0, 0xD20C, 0xD238, 0xD264] # PP 2, 3, 4, are + 1, 2, 3
    LEVEL_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
    MAX_HP_ADDRESSES = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269] # 2 bytes
    ATTACK_ADDRESSES = [0xD18F, 0xD1BB, 0xD1E7, 0xD213, 0xD23F, 0xD26B] # 2 bytes
    DEFENSE_ADDRESSES = [0xD191, 0xD1BD, 0xD1E9, 0xD215, 0xD241, 0xD26D] # 2 bytes
    SPEED_ADDRESSES = [0xD193, 0xD1BF, 0xD1EB, 0xD217, 0xD243, 0xD26F] # 2 bytes
    SPECIAL_ADDRESSES = [0xD195, 0xD1C1, 0xD1ED, 0xD219, 0xD245, 0xD271] # 2 bytes
    NICKNAME_ADDRESSES = [0xD2B5, 0xD2C0, 0xD2CB, 0xD2D6, 0xD2E1, 0xD2EC] # 10 bytes

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


    OPPONENT_LEVEL_ADDRESSES = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]


    def __init__(self, pyboy):
        self.pyboy = pyboy

    # Read functions for easier access to memory and parsing

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bcd(self, addr):
        num = self.read_m(addr)
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

    def read_multi_bcd(self, start_addr, length):
        return sum(self.read_bcd(start_addr + i) * 100 ** i for i in range(length))

    def read_multi_byte(self, start_addr, length):
        return sum(self.read_m(start_addr + i) * 256 ** i for i in range(length))

    def read_triple(self, start_add):
        return self.read_mulit_byte(start_add, 3)

    @staticmethod
    def bit_count(bits):
        return bin(bits).count('1')

    # New compact read, based on the dict with the stats
    def get_pokemion_info(self, info, pokemon_index, info_index=0):
        if not info in self.POKEMON_STATS:
            print(f"Error: {info} is not a valid pokemon info")
            return None

        pokemon_offset = self.POKEMON_OFFSET * pokemon_index

        stat_address = self.POKEMON_STATS[info]["address"] + pokemon_offset
        stat_length = self.POKEMON_STATS[info]["length"]
        if info_index > 0 and "amount" in self.POKEMON_STATS[info]:
            stat_address += info_index * stat_length
        
        return self.read_multibyte(stat_address, stat_length)



    # General stats

    def read_money(self):
        money = self.read_mulit_bc(self.MONEY_ADDRESS, 3)
        return money

    # Pokemon specific read functions 

    def read_current_hp(self, pokemon_index):
        return self.read_multibyte(self.HP_ADDRESSES[pokemon_index], 2)

    def read_max_hp(self, pokemon_index):
        return self.read_multibyte(self.HP_ADDRESSES[pokemon_index], 2)

    def read_level(self, pokemon_index):
        return self.read_m(self.LEVEL_ADDRESSES[pokemon_index])

    def read_attack(self, pokemon_index):
        return self.read_multibyte(self.ATTACK_ADDRESSES[pokemon_index], 2)


    def read_hp(self, start):
        return self.read_multi_byte(start, 2)

    def read_hp_fraction(self):
        hp_sum = sum(self.read_hp(addr) for addr in self.HP_ADDRESSES)
        max_hp_sum = sum(self.read_hp(addr) for addr in self.MAX_HP_ADDRESSES)
        return hp_sum / max_hp_sum


    def get_levels_sum(self):
        poke_levels = [max(self.read_m(addr) - 2, 0) for addr in self.LEVEL_ADDRESSES]
        return max(sum(poke_levels) - 4, 0)

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in self.PARTY_ADDRESSES]

    def read_opponent_levels(self):
        return [self.read_m(addr) for addr in self.OPPONENT_LEVEL_ADDRESSES]

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

    stats = {
        "HP Fraction": reader.read_hp_fraction(),
        "Money": reader.read_money(),
        "Levels Sum": reader.get_levels_sum(),
        "Badges": reader.get_badges(),
        "Party": reader.read_party(),
        "Opponent Levels": reader.read_opponent_levels()
    }

    print("Pokemon Red Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    args = parse_arguments()
    scan_saved_state(args.game_file_path, args.state_file_path)

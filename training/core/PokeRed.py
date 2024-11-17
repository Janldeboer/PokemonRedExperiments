import json
import numpy as np

from pyboy import PyBoy
from pyboy.utils import WindowEvent

ADDRESSES_FILE = "../core/poke_red_addresses.json"

class PokeRed:

    POKEMON_OFFSET = 0x002C
    OPPONENT_OFFSET = 0x0739

    VALID_ACTIONS = [
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
#        WindowEvent.PRESS_BUTTON_START,
        WindowEvent.PASS,
    ]

    RELEASE_ACTIONS = [
        WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.RELEASE_BUTTON_B,
#        WindowEvent.RELEASE_BUTTON_START,
    ]

    def __init__(
        self,
        gb_path,
        state_file=None,
        head="headless",
        hide_window=False,
        tick_callback=None,
    ):
        
        with open(ADDRESSES_FILE, "r") as f:
            self.STATS = json.load(f)
        
        self.pyboy = PyBoy(
            gb_path,
            window = "null" if hide_window else "SDL2",  # Use "null" for headless mode
        )

        self.action_freq = 24
        self.tick_callback = tick_callback

        if not head == "headless":
            self.pyboy.set_emulation_speed(6)

        if state_file:
            self.load_from_state(state_file)

    def load_from_state(self, state_file):
        with open(state_file, "rb") as f:
            self.pyboy.load_state(f)
            #print(f"Loaded state from {state_file}")

    def get_stat(self, info_name, pokemon_index=0, info_index=0, opponent=False):
        if info_name not in self.STATS:
            print(f"Error: {info_name} is not a valid stat")
            return None

        if not self.STATS[info_name]["is_poke_stat"] and (
            pokemon_index > 0 or opponent
        ):
            print(f"Error: {info_name} is not a valid pokemon info")
            return None

        if info_index >= self.STATS[info_name]["amount"]:
            print(
                f"Error: {info_name} does not have {info_index + 1} amount of {info_name} (index {info_index})"
            )
            return None

        stat_length = self.STATS[info_name]["length"]

        stat_address = self.STATS[info_name]["address"]
        stat_address += info_index * stat_length
        stat_address += self.POKEMON_OFFSET * pokemon_index
        stat_address += self.OPPONENT_OFFSET * opponent

        return self.read_multi_byte(stat_address, stat_length)

    def get_screen(self):
        return np.array(self.pyboy.screen)

    def get_poke_info(self, info, info_index=0, opponent=False):
        if info not in self.STATS:
            print(f"Error: {info} is not a valid pokemon info")
            return None
        return [
            self.get_stat(
                info, pokemon_index=i, info_index=info_index, opponent=opponent
            )
            for i in range(6)
        ]

    def get_agent_stats(self):
        """Convenience function to get all agent stats at once"""
        agent_stats = {
            "x": self.get_stat("X"),
            "y": self.get_stat("Y"),
            "map": self.get_stat("Map"),
            "pcount": self.get_stat("Party Count"),
            "levels": self.get_poke_info("Level"),
            "ptypes": [self.get_stat("Party", info_index=i) for i in range(6)],
            "Relative HP": self.read_hp_fraction(),
            "badge": self.get_stat("Badges"),
        }
        return agent_stats

    def custom_stats_1(self):
        """Custom stat set for CnnPolicy:
        For each of the players pokemon: HP, Level
        """

    def get_all_stats(self):
        all_stats = {}
        for stat, stat_info in self.STATS.items():
            if stat_info["is_poke_stat"]:
                all_stats[stat] = self.get_poke_info(stat)
            elif stat_info["amount"] > 1:
                all_stats[stat] = [
                    self.get_stat(stat, info_index=i)
                    for i in range(stat_info["amount"])
                ]
            else:
                all_stats[stat] = self.get_stat(stat)

        # TODO: remove this, but it's still used a lot
        all_stats["Relative HP"] = self.read_hp_fraction()

        return all_stats

    # For some reason we use this a lot
    def read_hp_fraction(self):
        hp_sum = sum(self.get_poke_info("HP"))
        max_hp_sum = sum(self.get_poke_info("Max HP"))
        return hp_sum / max_hp_sum

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.VALID_ACTIONS[action])
        for i in range(self.action_freq):
            if i == 8 and action < len(self.VALID_ACTIONS) - 1:
                self.pyboy.send_input(self.RELEASE_ACTIONS[action])
            self.pyboy.tick()
            if self.tick_callback:
                self.tick_callback()

        stats = self.get_all_stats()
        frame = self.get_screen()
        return stats, frame

    # Memory reading wrappers

    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def read_multi_byte(self, start_addr, length):
        return sum(self.read_m(start_addr + i) * 256**i for i in range(length))


def poke_red_demo(gb_path, state_path):
    poke_red = PokeRed(gb_path, state_path, head="headless")
    stats = poke_red.get_all_stats()
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    poke_red_demo("../PokemonRed.gb", "../../states/has_pokedex_nballs.state")

__all__ = ["PokeRed"]

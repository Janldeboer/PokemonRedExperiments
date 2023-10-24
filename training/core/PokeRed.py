import json

from pyboy import PyBoy
from pyboy.utils import WindowEvent


class PokeRed:
    STATS = {
        "X": {
            "address": 0xD362,
            "length": 1,
            "type": "int",
            "is_poke_stat": False,
            "amount": 1,
        },  #
        "Y": {
            "address": 0xD361,
            "length": 1,
            "type": "int",
            "is_poke_stat": False,
            "amount": 1,
        },  #
        "Map": {
            "address": 0xD35E,
            "length": 1,
            "type": "int",
            "is_poke_stat": False,
            "amount": 1,
        },  #
        "Party Count": {
            "address": 0xD163,
            "length": 1,
            "type": "int",
            "is_poke_stat": False,
            "amount": 1,
        },  # ?
        "Badges": {
            "address": 0xD356,
            "length": 1,
            "type": "int",
            "is_poke_stat": False,
            "amount": 1,
        },  #
        "Party": {
            "address": 0xD164,
            "length": 1,
            "type": "int",
            "is_poke_stat": False,
            "amount": 6,
        },  # ?
        "Pokemon": {
            "address": 0xD16B,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },  #
        "HP": {
            "address": 0xD16C,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },  #
        "Status": {
            "address": 0xD16F,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },  #
        "Type": {
            "address": 0xD170,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 2,
        },  #
        "Move": {
            "address": 0xD173,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 4,
        },  # ?
        "XP": {
            "address": 0xD179,
            "length": 4,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "HP EV": {
            "address": 0xD17C,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Attack EV": {
            "address": 0xD17E,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Defense EV": {
            "address": 0xD180,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Speed EV": {
            "address": 0xD182,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Special EV": {
            "address": 0xD184,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Attack/Defense IV": {
            "address": 0xD186,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Speed/Special IV": {
            "address": 0xD187,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "PP": {
            "address": 0xD188,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 4,
        },  # ?
        "Level": {
            "address": 0xD18C,
            "length": 1,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Max HP": {
            "address": 0xD18D,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Attack": {
            "address": 0xD18F,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Defense": {
            "address": 0xD191,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Speed": {
            "address": 0xD193,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Special": {
            "address": 0xD195,
            "length": 2,
            "type": "int",
            "is_poke_stat": True,
            "amount": 1,
        },
        "Nickname": {
            "address": 0xD2B5,
            "length": 10,
            "type": "string",
            "is_poke_stat": True,
            "amount": 1,
        },
    }

    POKEMON_OFFSET = 0x002C
    OPPONENT_OFFSET = 0x0739

    VALID_ACTIONS = [
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
        WindowEvent.PRESS_BUTTON_START,
        WindowEvent.PASS,
    ]

    RELEASE_ACTIONS = [
        WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.RELEASE_BUTTON_B,
        WindowEvent.RELEASE_BUTTON_START,
    ]

    def __init__(
        self,
        gb_path,
        state_file=None,
        head="headless",
        hide_window=False,
        tick_callback=None,
    ):
        self.pyboy = PyBoy(
            gb_path,
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window=hide_window,
        )

        self.action_freq = 24
        self.tick_callback = tick_callback

        self.pyboy.set_emulation_speed(0 if head == "headless" else 6)

        if state_file:
            self.load_from_state(state_file)

    def load_from_state(self, state_file):
        with open(state_file, "rb") as f:
            self.pyboy.load_state(f)
            print(f"Loaded state from {state_file}")

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
        return self.pyboy.botsupport_manager().screen().screen_ndarray()

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
            if i == 8 and action < 7:
                self.pyboy.send_input(self.RELEASE_ACTIONS[action])
            self.pyboy.tick()
            if self.tick_callback:
                self.tick_callback()

        stats = self.get_all_stats()
        frame = self.get_screen()
        return stats, frame

    # Memory reading wrappers

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_multi_byte(self, start_addr, length):
        return sum(self.read_m(start_addr + i) * 256**i for i in range(length))


def poke_red_demo(gb_path, state_path):
    poke_red = PokeRed(gb_path, state_path, head="headless")
    stats = poke_red.get_all_stats()
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    poke_red_demo("../PokemonRed.gb", "../has_pokedex_nballs.state")

__all__ = ["PokeRed"]

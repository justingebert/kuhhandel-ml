import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlattenAuctionActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # store original space if you want
        self.original_action_space = env.action_space  # Dict(...)

        # build flat MultiDiscrete space
        self.n_bid_bits = 33
        self.n_offer_bits = 33
        self.n_counter_bits = 33

        nvec = np.array(
            [
                2,   # auction_or_trade
                2,   # auction_pass
                2,   # auction_buy_or_sell_as_auctioneer
                3,   # trade_enemy
                10,  # trade_animal
                *([2] * self.n_bid_bits),     # auction_bid
                *([2] * self.n_offer_bits),   # trade_offer
                *([2] * self.n_counter_bits), # trade_counter
            ],
            dtype=np.int64,
        )

        self.action_space = spaces.MultiDiscrete(nvec)

    def action(self, action: np.ndarray):
        """
        Convert flat MultiDiscrete action from SB3 → nested Dict action expected by underlying env.
        """
        action = np.asarray(action, dtype=np.int64)

        idx = 0
        auction_or_trade = int(action[idx]); idx += 1
        auction_pass = int(action[idx]); idx += 1
        auction_buy_or_sell = int(action[idx]); idx += 1
        trade_enemy = int(action[idx]); idx += 1
        trade_animal = int(action[idx]); idx += 1

        auction_bid = action[idx : idx + self.n_bid_bits].astype(np.int8)
        idx += self.n_bid_bits
        trade_offer = action[idx : idx + self.n_offer_bits].astype(np.int8)
        idx += self.n_offer_bits
        trade_counter = action[idx : idx + self.n_counter_bits].astype(np.int8)

        nested_action = {
            "auction_or_trade": auction_or_trade,
            "auction": {
                "auction_bid": auction_bid,
                "auction_pass": auction_pass,
                "auction_buy_or_sell_as_auctioneer": auction_buy_or_sell,
            },
            "trade": {
                "trade_enemy": trade_enemy,
                "trade_animal": trade_animal,
                "trade_offer": trade_offer,
                "trade_counter": trade_counter,
            },
        }

        return nested_action

    # optional, if you ever need to log or sample nested actions from old agents
    def reverse_action(self, nested_action: dict) -> np.ndarray:
        a = []

        a.append(nested_action["auction_or_trade"])
        a.append(nested_action["auction"]["auction_pass"])
        a.append(nested_action["auction"]["auction_buy_or_sell_as_auctioneer"])
        a.append(nested_action["trade"]["trade_enemy"])
        a.append(nested_action["trade"]["trade_animal"])

        a.extend(nested_action["auction"]["auction_bid"].tolist())
        a.extend(nested_action["trade"]["trade_offer"].tolist())
        a.extend(nested_action["trade"]["trade_counter"].tolist())

        return np.array(a, dtype=np.int64)

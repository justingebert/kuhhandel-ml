
import unittest
import numpy as np
from gameengine.game import Game, GamePhase
from gameengine.actions import Actions, ActionType
from rl.env import KuhhandelEnv, MONEY_UNKNOWN

class TestKnownMoney(unittest.TestCase):

    def setUp(self):
        self.env = KuhhandelEnv(num_players=3)
        self.env.reset(seed=42)
        # Ensure initial state matches assumption (everyone has money)
        self.assertTrue(all(len(p.money) > 0 for p in self.env.game.players))

    def test_initial_knowledge(self):
        """Test that initially everyone knows everyone's money."""
        obs = self.env.get_observation_for_player(0)
        known_money = obs["known_player_money"]
        
        # Player 0 should know everyone's money (including themselves)
        for i in range(3):
            # Known value should match actual value DIVIDED BY 10 (MONEY_STEP)
            expected = self.env.game.players[i].get_total_money() // 10
            self.assertEqual(known_money[i], expected)
            self.assertNotEqual(known_money[i], MONEY_UNKNOWN)

    def test_auction_maintains_knowledge(self):
        """Test that auctions (public) preserve knowledge."""
        # Force an auction scenario
        # Player 0 starts auction
        # Let's interact with game directly and call _update_money_beliefs manually
        # Ensure deck isn't empty (reset puts 40 cards)
        self.assertTrue(len(self.env.game.animal_deck) > 0)
        
        self.env.game.start_auction()
        
        # Ensure an animal was drawn (auction started)
        self.assertIsNotNone(self.env.game.current_animal)
        
        self.env.game.place_bid(1, 10) # Player 1 bids 10
        self.env.game.end_auction_bidding()
        self.env.game.current_player_idx = 0 # Auctioneer
        self.env.game.auctioneer_passes() # Auctioneer accepts bit, P1 gets animal, P1 pays P0
        
        # Manually trigger update
        self.env._update_money_beliefs()
        
        obs0 = self.env.get_observation_for_player(0) # P0 (Receiver)
        obs1 = self.env.get_observation_for_player(1) # P1 (Payer)
        obs2 = self.env.get_observation_for_player(2) # P2 (Observer)
        
        # P2 should still know P0 and P1's money because auction is public
        self.assertNotEqual(obs2["known_player_money"][0], MONEY_UNKNOWN)
        self.assertNotEqual(obs2["known_player_money"][1], MONEY_UNKNOWN)
        
        # Check values reflect payment (approximate check of logic preservation)
        expected = self.env.game.players[1].get_total_money() // 10
        self.assertEqual(obs2["known_player_money"][1], expected)

    def test_trade_hides_knowledge(self):
        """Test that cow trades hide knowledge from third parties."""
        # Setup: P1 trades with P2. P0 is observer.
        # We need P1 and P2 to have matching animals.
        # Let's force-give them animals.
        from gameengine.Animal import AnimalType
        self.env.game.players[1].add_animal(self.env.game.animal_deck[0]) # Give P1 some animal
        # We need to find a match. Actually, we just need to TRIGGER the log events.
        # We can fake the log!
        
        self.env.game._log_action("start_cow_trade", {
            "initiator": 1,
            "target": 2,
            "animal": "Donkey",
            "offer": 10,
            "counter_offer": 0
        })
        self.env.game._log_action("resolve_trade", {
            "winner": "initiator",
            "offer": 10,
            "counter": 0,
            "animals_transferred": 1
        })
        
        self.env._update_money_beliefs()
        
        obs0 = self.env.get_observation_for_player(0) # P0 (Observer)
        obs1 = self.env.get_observation_for_player(1) # P1 (Initiator)
        
        # P0 should NOT know P1 or P2 anymore
        self.assertEqual(obs0["known_player_money"][1], MONEY_UNKNOWN)
        self.assertEqual(obs0["known_player_money"][2], MONEY_UNKNOWN)
        # P0 should still know themselves
        self.assertNotEqual(obs0["known_player_money"][0], MONEY_UNKNOWN)
        
        # P1 (involved) SHOULD still know P2 (involved) - Logic says "Involved keep known"
        # My implementation: 
        # "Everyone NOT p1 or p2 loses knowledge of p1 and p2"
        # So p1 IS p1, so p1 does NOT lose knowledge of p1?
        # Wait: "self.money_known[obs_id, p1_id] = False"
        # My loop: "if obs_id != p1_id and obs_id != p2_id:" -> Then mask.
        # So P1 (obs_id=1) is != P1? False. So P1 keeps knowing P1.
        # P1 (obs_id=1) is != P2? True. AND P1 != P1 (False).
        # Boolean logic: (obs != p1 AND obs != p2).
        # For P1: (False AND True) -> False. P1 does NOT lose knowledge. Correct.
        
        self.assertNotEqual(obs1["known_player_money"][2], MONEY_UNKNOWN)


    def test_zero_cards_knowledge_restoration(self):
        """Test that 0 cards restores knowledge."""
        # First, mask P1's money via cheat
        self.env.money_known[:, 1] = False
        
        obs0 = self.env.get_observation_for_player(0)
        self.assertEqual(obs0["known_player_money"][1], MONEY_UNKNOWN)
        
        # Now empty P1's money
        self.env.game.players[1].money = []
        
        # Trigger update (normally happens in step)
        self.env._update_money_beliefs()
        
        obs0 = self.env.get_observation_for_player(0)
        # Should be known now (and equal to 0)
        self.assertEqual(obs0["known_player_money"][1], 0)
        self.assertNotEqual(obs0["known_player_money"][1], MONEY_UNKNOWN)

if __name__ == '__main__':
    unittest.main()

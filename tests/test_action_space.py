"""
Tests for shared money action space reduction
"""
import pytest
import numpy as np
from rl.env import KuhhandelEnv, N_ACTIONS, ACTION_MONEY_SHARED_BASE, ACTION_MONEY_SHARED_END
from gameengine import Game, GamePhase
from gameengine.actions import Actions


class TestActionSpaceSize:
    """Test that action space has been reduced correctly."""
    
    def test_action_space_size(self):
        """Verify new action space is 497 (down from 1,439)."""
        env = KuhhandelEnv()
        print(f"DEBUG: env.action_space.n = {env.action_space.n}")
        print(f"DEBUG: N_ACTIONS = {N_ACTIONS}")
        assert env.action_space.n == 497
        assert N_ACTIONS == 497
    
    def test_money_shared_range(self):
        """Verify shared money action range."""
        # Should have 471 money levels (0 to 470)
        expected_money_actions = 471
        actual_money_actions = ACTION_MONEY_SHARED_END - ACTION_MONEY_SHARED_BASE + 1
        assert actual_money_actions == expected_money_actions


class TestSharedMoneyDecoding:
    """Test that money actions decode correctly based on game phase."""
    
    def test_auction_bidding_pass(self):
        """Test that money level 0 decodes to pass in auction."""
        env = KuhhandelEnv()
        env.reset()
        
        # Start auction to enter AUCTION_BIDDING phase
        env.game.start_auction()
        assert env.game.phase == GamePhase.AUCTION_BIDDING
        
        # Money level 0 should decode to pass
        action_idx = ACTION_MONEY_SHARED_BASE  # Level 0
        decoded = env.decode_action(action_idx, env.game)
        assert decoded.type.name == "AUCTION_PASS"
    
    def test_auction_bidding_bid(self):
        """Test that money levels >0 decode to bid in auction."""
        env = KuhhandelEnv()
        env.reset()
        
        env.game.start_auction()
        
        # Money level 10 should decode to bid(100)
        action_idx = ACTION_MONEY_SHARED_BASE + 10
        decoded = env.decode_action(action_idx, env.game)
        assert decoded.type.name == "AUCTION_BID"
        assert decoded.amount == 100
    
    def test_cow_trade_offer(self):
        """Test that money actions decode to offer in COW_TRADE_OFFER phase."""
        env = KuhhandelEnv()
        env.reset()
        
        # Manually set to COW_TRADE_OFFER phase
        env.game.phase = GamePhase.COW_TRADE_OFFER
        env.game.trade_initiator = 0
        
        # Money level 20 should decode to offer(200)
        action_idx = ACTION_MONEY_SHARED_BASE + 20
        decoded = env.decode_action(action_idx, env.game)
        assert decoded.type.name == "COW_TRADE_OFFER"
        assert decoded.amount == 200
    
    def test_cow_trade_counter(self):
        """Test that money actions decode to counter in COW_TRADE_RESPONSE phase."""
        env = KuhhandelEnv()
        env.reset()
        
        # Manually set to COW_TRADE_RESPONSE phase
        env.game.phase = GamePhase.COW_TRADE_RESPONSE
        env.game.trade_target = 0
        
        # Money level 15 should decode to counter_offer(150)
        action_idx = ACTION_MONEY_SHARED_BASE + 15
        decoded = env.decode_action(action_idx, env.game)
        assert decoded.type.name == "COUNTER_OFFER"
        assert decoded.amount == 150


class TestActionMasking:
    """Test that action masking works correctly with shared money space."""
    
    def test_no_money_actions_in_turn_choice(self):
        """Money actions should not be valid during PLAYER_TURN_CHOICE."""
        env = KuhhandelEnv()
        env.reset()
        
        # Should be in PLAYER_TURN_CHOICE phase
        assert env.game.phase == GamePhase.PLAYER_TURN_CHOICE
        
        mask = env.get_action_mask()
        
        # No money actions should be valid
        money_mask = mask[ACTION_MONEY_SHARED_BASE:ACTION_MONEY_SHARED_END+1]
        assert np.sum(money_mask) == 0
    
    def test_pass_enabled_in_auction(self):
        """Pass (money level 0) should be enabled during auction."""
        env = KuhhandelEnv()
        env.reset()
        
        # Start auction
        env.game.start_auction()
        
        # Get mask for a non-auctioneer player
        bidder_id = (env.game.current_player_idx + 1) % 3
        mask = env.get_action_mask_for_player(bidder_id)
        
        # Pass (level 0) should be valid
        assert mask[ACTION_MONEY_SHARED_BASE] == 1
    
    def test_valid_bids_masked_correctly(self):
        """Only bids within player's money and above high bid should be valid."""
        env = KuhhandelEnv()
        env.reset()
        
        # Set up auction scenario
        env.game.start_auction()
        env.game.auction_high_bid = 50  # Current high bid
        
        bidder_id = (env.game.current_player_idx + 1) % 3
        player_money = env.game.players[bidder_id].get_total_money()
        
        mask = env.get_action_mask_for_player(bidder_id)
        
        # Bids below or equal to 50 should be invalid (except pass at 0)
        for level in range(1, 6):  # 10, 20, 30, 40, 50
            action_idx = ACTION_MONEY_SHARED_BASE + level
            assert mask[action_idx] == 0, f"Bid {level*10} should be invalid"
        
        # Bids above 50 and within money should be valid
        min_valid_level = 6  # 60
        max_valid_level = player_money // 10
        
        for level in range(min_valid_level, min(max_valid_level + 1, 471)):
            action_idx = ACTION_MONEY_SHARED_BASE + level
            if action_idx <= ACTION_MONEY_SHARED_END:
                assert mask[action_idx] == 1, f"Bid {level*10} should be valid"


class TestFullGameIntegration:
    """Test that a full game can be played with new action space."""
    
    def test_full_game_with_random_actions(self):
        """Play a full game using random valid actions."""
        env = KuhhandelEnv()
        obs, _ = env.reset()
        
        done = False
        steps = 0
        max_steps = 500
        
        while not done and steps < max_steps:
            # Get valid actions
            mask = env.get_action_mask()
            valid_actions = np.where(mask == 1)[0]
            
            # Should always have at least one valid action
            assert len(valid_actions) > 0, f"No valid actions at step {steps}, phase {env.game.phase}"
            
            # Choose random valid action
            action = np.random.choice(valid_actions)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            if done or truncated:
                break
        
        # Game should complete
        assert done or truncated, f"Game did not complete in {max_steps} steps"
        print(f"✓ Game completed successfully in {steps} steps")


def test_action_space_constants():
    """Verify that action space constants are correct."""
    from rl.env import (
        ACTION_START_AUCTION,
        ACTION_COW_CHOOSE_OPP_BASE,
        ACTION_COW_CHOOSE_OPP_END,
        ACTION_AUCTIONEER_ACCEPT,
        ACTION_AUCTIONEER_BUY,
        ACTION_COW_CHOOSE_ANIMAL_BASE,
        ACTION_COW_CHOOSE_ANIMAL_END,
        ACTION_MONEY_SHARED_BASE,
        ACTION_MONEY_SHARED_END,
        ACTION_COW_BLUFF_BASE,
        ACTION_COW_BLUFF_END,
        N_ACTIONS,
    )
    
    # Verify sequence
    assert ACTION_START_AUCTION == 0
    assert ACTION_COW_CHOOSE_OPP_BASE == 1
    assert ACTION_COW_CHOOSE_OPP_END == 2
    assert ACTION_AUCTIONEER_ACCEPT == 3
    assert ACTION_AUCTIONEER_BUY == 4
    assert ACTION_COW_CHOOSE_ANIMAL_BASE == 5
    assert ACTION_COW_CHOOSE_ANIMAL_END == 14
    assert ACTION_MONEY_SHARED_BASE == 15
    assert ACTION_MONEY_SHARED_END == 485
    assert ACTION_COW_BLUFF_BASE == 486
    assert ACTION_COW_BLUFF_END == 496
    assert N_ACTIONS == 497


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running smoke tests...")
    
    print("\n1. Testing action space size...")
    test = TestActionSpaceSize()
    test.test_action_space_size()
    print("✓ Action space is 497")
    
    print("\n2. Testing money action decoding...")
    test2 = TestSharedMoneyDecoding()
    test2.test_auction_bidding_bid()
    test2.test_cow_trade_offer()
    test2.test_cow_trade_counter()
    print("✓ Money actions decode correctly")
    
    print("\n3. Testing action masking...")
    test3 = TestActionMasking()
    test3.test_no_money_actions_in_turn_choice()
    test3.test_pass_enabled_in_auction()
    print("✓ Action masking works")
    
    print("\n4. Testing full game integration...")
    test4 = TestFullGameIntegration()
    test4.test_full_game_with_random_actions()
    
    print("\n✅ All smoke tests passed!")

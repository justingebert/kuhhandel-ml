class RewardConfig:
    """Base configuration for reward calculation"""

    # Vulnerability penalty
    no_money_penalty = -0.2 #per round

    # Money dominance
    money_dominance_bonus = 0.05 #per round

    # Auction wins
    high_bidder_wins_reward = 0.1
    auctioneer_gets_free_reward = 0.2
    auctioneer_gets_free_penalty = -0.2 #to everyone else
    auctioneer_buys_reward = 0.1 #for money reciever
    auctioneer_self_buy_penalty = -0.05 #for auctioneer

    # Bidding behavior
    self_overbid_penalty = -0.1
    overbid100_penelty = 1.0  # for 100 above highbid

    # Quartets
    early_quartet_bonus = 0.5

    # Cow trades
    trade_win_base_reward = 0.1
    trade_win_efficiency_bonus = 0.1
    trade_loss_base_penalty = -0.1
    trade_loss_money_compensation = 0.15
    trade_loss_during_auction_penalty = -0.1

    # Game end
    win_reward = 2


class RewardMinimalAggressiveConfig(RewardConfig):
    # Vulnerability penalty
    no_money_penalty = 0 #per round

    # Money dominance
    money_dominance_bonus = 0 #per round

    # Auction wins
    high_bidder_wins_reward = 0.2
    auctioneer_gets_free_reward = 0.2
    auctioneer_gets_free_penalty = -0.2 #to everyone else
    auctioneer_buys_reward = 0.2 #for money reciever
    auctioneer_self_buy_penalty = 0 #for auctioneer

    # Bidding behavior
    self_overbid_penalty = 0
    overbid100_penelty = 0

    # Quartets
    early_quartet_bonus = 1

    # Cow trades
    trade_win_base_reward = 0.1
    trade_win_efficiency_bonus = 0.1
    trade_loss_base_penalty = -0.1
    trade_loss_money_compensation = 0.15
    trade_loss_during_auction_penalty = -0.1

    # Game end
    win_reward = 2


class WinOnlyConfig(RewardConfig):
    """Base configuration for reward calculation"""

    # Vulnerability penalty
    no_money_penalty = 0 #per round

    # Money dominance
    money_dominance_bonus = 0 #per round

    # Auction wins
    high_bidder_wins_reward = 0
    auctioneer_gets_free_reward = 0
    auctioneer_gets_free_penalty = 0 #to everyone else
    auctioneer_buys_reward = 0 #for money reciever
    auctioneer_self_buy_penalty = 0 #for auctioneer

    # Bidding behavior
    self_overbid_penalty = 0
    overbid100_penelty = 0

    # Quartets
    early_quartet_bonus = 0.5

    # Cow trades
    trade_win_base_reward = 0
    trade_win_efficiency_bonus = 0
    trade_loss_base_penalty = 0
    trade_loss_money_compensation = 0
    trade_loss_during_auction_penalty = 0

    # Game end
    win_reward = 2

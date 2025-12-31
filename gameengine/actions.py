"""Type-safe action definitions for the game using dataclasses."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Union
from .Animal import AnimalType

class ActionType(Enum):
    """Types of actions players can take."""
    START_AUCTION = "start_auction"
    START_COW_TRADE = "start_cow_trade"
    AUCTION_BID = "auction-bid"
    AUCTION_PASS = "pass"
    BUY_AS_AUCTIONEER = "buy_as_auctioneer"
    PASS_AS_AUCTIONEER = "pass_as_auctioneer"
    COW_TRADE_CHOOSE_OPPONENT = "cow_trade_choose_opponent"
    COW_TRADE_CHOOSE_ANIMAL = "cow_trade_choose_animal"
    COW_TRADE_OFFER = "cow_trade_offer"
    COW_TRADE_ADD_BLUFF = "cow_trade_add_bluff"
    COUNTER_OFFER = "counter_offer"

# Base class for all actions (optional but recommended)
@dataclass(frozen=True)
class GameActionBase:
    """Base class for all game actions."""
    pass

    def __str__(self):
        if hasattr(self, "_template"):
            return self._template.format(s=self)
        return super().__str__()

@dataclass(frozen=True)
class StartAuctionAction(GameActionBase):
    type: ActionType = field(default=ActionType.START_AUCTION, init=False)

    _template = "Start Auction"

@dataclass(frozen=True)
class AuctionBidAction(GameActionBase):
    amount: int
    type: ActionType = field(default=ActionType.AUCTION_BID, init=False)

    _template = "Bid {s.amount}"

@dataclass(frozen=True)
class AuctionPassAction(GameActionBase):
    type: ActionType = field(default=ActionType.AUCTION_PASS, init=False)

    _template = "Pass"

@dataclass(frozen=True)
class BuyAsAuctioneerAction(GameActionBase):
    type: ActionType = field(default=ActionType.BUY_AS_AUCTIONEER, init=False)

    _template = "Buy (Auctioneer)"

@dataclass(frozen=True)
class PassAsAuctioneerAction(GameActionBase):
    type: ActionType = field(default=ActionType.PASS_AS_AUCTIONEER, init=False)

    _template = "Pass (Auctioneer)"

@dataclass(frozen=True)
class CowTradeChooseOpponentAction(GameActionBase):
    target_id: int
    type: ActionType = field(default=ActionType.COW_TRADE_CHOOSE_OPPONENT, init=False)

    _template = "Start Trade with Player {s.target_id}"

@dataclass(frozen=True)
class CowTradeChooseAnimalAction(GameActionBase):
    animal_type: AnimalType
    type: ActionType = field(default=ActionType.COW_TRADE_CHOOSE_ANIMAL, init=False)

    _template = "Choice Animal: {s.animal_type.display_name}"

@dataclass(frozen=True)
class CowTradeOfferAction(GameActionBase):
    amount: int
    type: ActionType = field(default=ActionType.COW_TRADE_OFFER, init=False)

    _template = "Offer {s.amount}"

@dataclass(frozen=True)
class CowTradeAddBluffAction(GameActionBase):
    amount: int # number of 0 cards to add
    type: ActionType = field(default=ActionType.COW_TRADE_ADD_BLUFF, init=False)

    _template = "Add Bluff ({s.amount} zero-cards)"

@dataclass(frozen=True)
class CowTradeAddBluffAction(GameActionBase):
    amount: int # number of 0 cards to add
    type: ActionType = field(default=ActionType.COW_TRADE_ADD_BLUFF, init=False)

@dataclass(frozen=True)
class CowTradeCounterOfferAction(GameActionBase):
    amount: int
    type: ActionType = field(default=ActionType.COUNTER_OFFER, init=False)

    _template = "Counter Offer {s.amount}"


# Union of all possible actions
GameAction = Union[
    StartAuctionAction,
    AuctionBidAction,
    AuctionPassAction,
    PassAsAuctioneerAction,
    BuyAsAuctioneerAction,
    CowTradeChooseOpponentAction,
    CowTradeChooseAnimalAction,
    CowTradeOfferAction,
    CowTradeAddBluffAction,
    CowTradeCounterOfferAction
]


class Actions:
    """Factory class for creating action objects with a clean API."""

    @staticmethod
    def start_auction() -> StartAuctionAction:
        """Create a start auction action."""
        return StartAuctionAction()

    @staticmethod
    def bid(amount: int) -> AuctionBidAction:
        """Create a bid action."""
        return AuctionBidAction(amount=amount)

    @staticmethod
    def pass_action() -> AuctionPassAction:
        """Create a pass action."""
        return AuctionPassAction()
    
    @staticmethod
    def buy_as_auctioneer() -> BuyAsAuctioneerAction:
        """Create a buy as auctioneer action."""
        return BuyAsAuctioneerAction()
    
    @staticmethod
    def pass_as_auctioneer() -> PassAsAuctioneerAction:
        """Create a pass as auctioneer action."""
        return PassAsAuctioneerAction()

    @staticmethod
    def cow_trade_choose_opponent(target_id: int) -> CowTradeChooseOpponentAction:
        """Create a cow trade choose opponent action."""
        return CowTradeChooseOpponentAction(target_id=target_id)

    @staticmethod
    def cow_trade_choose_animal(animal_type: AnimalType) -> CowTradeChooseAnimalAction:
        """Create a cow trade choose animal action."""
        return CowTradeChooseAnimalAction(animal_type=animal_type)

    @staticmethod
    def cow_trade_offer(amount: int) -> CowTradeOfferAction:
        """Create a cow trade offer action."""
        return CowTradeOfferAction(amount=amount)

    @staticmethod
    def cow_trade_add_bluff(amount: int) -> CowTradeAddBluffAction:
        """Create a cow trade add bluff action."""
        return CowTradeAddBluffAction(amount=amount)

    @staticmethod
    def counter_offer(amount: int) -> CowTradeCounterOfferAction:
        """Create a counter offer action."""
        return CowTradeCounterOfferAction(amount=amount)




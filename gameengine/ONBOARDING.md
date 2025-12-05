# 🎮 Welcome to Kuhhandel-ML! Your Onboarding Guide

Welcome aboard! This guide will help you understand how this project works, even if you're new to programming and Python. Think of this as your friendly tour guide through the codebase.

## 📚 Table of Contents

1. [What is this Project?](#what-is-this-project)
2. [The Big Picture](#the-big-picture)
3. [Project Structure](#project-structure)
4. [The Game Engine - How the Game Works](#the-game-engine)
5. [The Reinforcement Learning (RL) Layer](#the-reinforcement-learning-layer)
6. [How Everything Works Together](#how-everything-works-together)
7. [Getting Started - Running the Code](#getting-started)
8. [Key Concepts to Understand](#key-concepts)

---

## 🎯 What is this Project?

This project implements a card game called **Kuhhandel** (also known as "You're Bluffing") in Python. But it's not just a game - it's also a **machine learning research project** that teaches a computer to play the game using **Reinforcement Learning (RL)**.

Think of it like teaching a robot to play chess, except we're teaching it to play Kuhhandel!

---

## 🌍 The Big Picture

This project has **two main parts**:

```
┌─────────────────────────────────────────┐
│                                         │
│         YOUR PROJECT                    │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │                                 │   │
│  │    GAME ENGINE                  │   │
│  │  (The actual game logic)        │   │
│  │  • Rules                        │   │
│  │  • Players                      │   │
│  │  • Cards                        │   │
│  │  • Game flow                    │   │
│  │                                 │   │
│  └─────────────────────────────────┘   │
│            ↕                            │
│  ┌─────────────────────────────────┐   │
│  │                                 │   │
│  │    RL LAYER                     │   │
│  │  (Teaching AI to play)          │   │
│  │  • Environment                  │   │
│  │  • RL Agent                     │   │
│  │  • Training                     │   │
│  │                                 │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

Here's what each folder and file does:

```
kuhhandel-ml/
│
├── gameengine/              # The Game Engine (Part 1)
│   ├── Animal.py           # Defines animal cards (Cow, Horse, etc.)
│   ├── Money.py            # Defines money cards
│   ├── Player.py           # Represents a player
│   ├── game.py             # The main game logic (big file!)
│   ├── controller.py       # Controls the game flow
│   ├── agent.py            # Base class for any player (human or AI)
│   └── actions.py          # All possible actions players can take
│
├── rl/                     # The RL Layer (Part 2)
│   ├── env.py             # Gymnasium environment (RL wrapper)
│   └── rl_agent.py        # The RL agent that learns to play
│
├── tests/                  # Example code and tests
│   ├── demo_game.py       # Shows how to run a game
│   └── test_game.py       # Tests to verify game works correctly
│
├── pyproject.toml         # Project configuration & dependencies
├── Readme.md              # Brief project overview
└── ONBOARDING.md          # This file! 😊
```

---

## 🎮 The Game Engine

Let's dive into each component of the game engine. I'll explain them like building blocks.

### 🐄 Animal.py - The Animal Cards

**What it does:** Defines the types of animals in the game and their point values.

**Simple analogy:** Think of this as a catalog of trading cards. Each card shows an animal, and each set of 4 identical animals has a point value.

**Key components:**
- `AnimalType`: An enum (list of constants) with 10 animal types
  - Examples: `CHICKEN` (10 points), `COW` (800 points), `HORSE` (1000 points)
- `AnimalCard`: Represents a single card
  - Each card has an `animal_type` and a unique `card_id`

**Example:**
```python
# Creating a horse card
horse_card = AnimalCard(AnimalType.HORSE, card_id=1)
print(horse_card)  # Output: AnimalCard(Horse)
```

---

### 💰 Money.py - The Money System

**What it does:** Manages money cards that players use to bid and trade.

**Simple analogy:** Like having a wallet with different bills - $10, $50, $100, etc.

**Key components:**
- `MoneyCard`: A single money card with a `value` (0, 10, 50, 100, 200, or 500)
- `MoneyDeck`: Creates and manages all the money in the game
  - Has a specific distribution: 10 × $0, 20 × $10, 10 × $50, etc.
  - Gives starting money to each player: 2 × $0, 4 × $10, 1 × $50

**Example:**
```python
# Create the money deck
deck = MoneyDeck()

# Get starting money for a player
starting_money = deck.get_starting_money()
# Player gets: [0, 0, 10, 10, 10, 10, 50]
```

---

### 👤 Player.py - The Players

**What it does:** Represents a player in the game, tracking their money and animals.

**Simple analogy:** This is like your player profile in a video game - it knows what you own, your score, etc.

**What a player has:**
- `player_id`: A number (0, 1, 2, etc.)
- `name`: The player's name
- `money`: List of money cards they own
- `animals`: List of animal cards they've collected

**What a player can do:**
- `add_money()` / `remove_money()`: Get or spend money
- `add_animal()` / `remove_animals()`: Collect or trade animals
- `get_total_money()`: Count how much money they have
- `get_animal_counts()`: See how many of each animal they have
- `calculate_score()`: Calculate final points
  - **Important:** You only get points for **complete sets of 4** identical animals!

**Example:**
```python
player = Player(player_id=0, name="Alice")
player.add_animal(AnimalCard(AnimalType.COW, 1))
player.add_animal(AnimalCard(AnimalType.COW, 2))

print(player.get_animal_count(AnimalType.COW))  # Output: 2
print(player.has_complete_set(AnimalType.COW))  # Output: False (needs 4)
```

---

### 🎲 game.py - The Game Logic (The Brain!)

**What it does:** This is the **most important file** - it contains all the game rules and logic. It's like the referee in a sports game.

**Simple analogy:** Imagine a board game instruction manual that's actually alive and can enforce all the rules automatically.

**Key concepts:**

#### Game Phases
The game goes through different phases, like chapters in a book:

```python
class GamePhase(Enum):
    SETUP = "setup"              # Initial setup
    PLAYER_TURN = "player_turn"  # Player chooses auction or trade
    AUCTION = "auction"          # Players are bidding
    COW_TRADE = "cow_trade"      # Two players are trading
    GAME_OVER = "game_over"      # Game finished
```

#### Action Types
These are all the things a player can do:

```python
class ActionType(Enum):
    START_AUCTION = "start_auction"       # Start an auction
    START_COW_TRADE = "start_cow_trade"   # Start a trade with another player
    BID = "bid"                           # Bid money in an auction
    PASS = "pass"                         # Pass/decline
    BUY_AS_AUCTIONEER = "buy_as_auctioneer"  # Auctioneer buys their own item
    ACCEPT_OFFER = "accept_offer"         # Accept a trade offer
    COUNTER_OFFER = "counter_offer"       # Make a counter-offer in trade
```

#### The Game Class
The `Game` class has many methods. Here are the most important ones:

| Method | What it does |
|--------|-------------|
| `setup()` | Prepares the game: creates cards, gives starting money |
| `get_valid_actions()` | Returns what actions are currently allowed |
| `start_auction()` | Draws an animal card and starts bidding |
| `place_bid()` | A player places a bid |
| `auctioneer_buys()` | Auctioneer chooses to buy at the highest bid |
| `auctioneer_passes()` | Auctioneer passes, highest bidder wins |
| `start_cow_trade()` | Player starts a trade with another player |
| `accept_trade_offer()` | Target player accepts the trade |
| `counter_trade_offer()` | Target player makes a counter-offer |
| `is_game_over()` | Checks if game is finished |
| `get_scores()` | Calculates final scores for all players |

**How a typical turn works:**

```
1. Player's turn starts
   ↓
2. Player chooses: Start Auction OR Start Cow Trade
   ↓
3a. If Auction:                    3b. If Cow Trade:
    - Draw animal card                 - Choose target player & animal type
    - Players bid                      - Make an offer with money cards
    - Auctioneer decides               - Target accepts or counter-offers
    - Winner gets animal               - Higher bidder wins animals
   ↓
4. Next player's turn
```

---

### 🎮 controller.py - The Game Controller

**What it does:** Manages the game flow and coordinates between the `Game` and the agents (players).

**Simple analogy:** Like a game master in Dungeons & Dragons - doesn't play, but keeps the game moving.

**Why we need it:** The `Game` class knows the rules, but the `Controller` actually runs the game by:
1. Asking agents for their actions
2. Telling the game to execute those actions
3. Repeating until the game ends

**Key methods:**

| Method | What it does |
|--------|-------------|
| `run()` | Runs the entire game from start to finish |
| `step()` | Executes one step (one decision) |
| `_handle_player_turn()` | Asks current player what they want to do |
| `_handle_auction_phase()` | Runs the entire auction process |
| `_handle_cow_trade_phase()` | Handles the trade negotiation |

**Example flow:**
```python
# Create a game
game = Game(num_players=3)
game.setup()

# Create agents (players)
agents = [RandomAgent("Bot 1"), RandomAgent("Bot 2"), RandomAgent("Bot 3")]

# Create controller
controller = GameController(game, agents)

# Run the game!
final_scores = controller.run(max_turns=500)
```

---

### 🤖 agent.py - The Agent Interface

**What it does:** Defines what any player (human or AI) must be able to do.

**Simple analogy:** It's like a job description. Anyone who wants to play the game must be able to follow this interface.

**The Agent interface:**
```python
class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.player_id = None
    
    @abstractmethod
    def get_action(self, game: Game, valid_actions: List[ActionType]) -> GameAction:
        """Must be implemented by any agent"""
        pass
```

**What this means:**
- `ABC` = Abstract Base Class (a template)
- Any agent must implement `get_action()` - this is where they decide what to do
- The controller calls this method when it's the agent's turn

---

### 🎯 actions.py - Action Definitions

**What it does:** Defines all possible actions as typed objects using dataclasses.

**Simple analogy:** Like creating specific forms for different types of requests. Each form has the exact fields needed.

**Why it's useful:** Instead of using messy dictionaries, we have clean, type-safe action objects.

**Example actions:**
```python
from gameengine.actions import Actions

# Create different types of actions
auction = Actions.start_auction()
bid = Actions.bid(amount=100)
trade = Actions.start_cow_trade(
    target_id=1,
    animal_type=AnimalType.COW,
    money_cards=[money_card1, money_card2]
)
```

---

## 🤖 The Reinforcement Learning Layer

Now let's look at how we teach an AI to play!

### 🏋️ env.py - The Gymnasium Environment

**What it does:** Wraps the game in a format that RL algorithms can understand.

**What is Gymnasium?** It's a Python library (like a toolkit) that provides a standard way to create game environments for RL. Think of it as a universal adapter.

**Why we need it:** RL algorithms (like PPO from Stable-Baselines3) expect a specific format:
- **Observation space**: What the agent can see (game state)
- **Action space**: What the agent can do (discrete actions)
- **Reset**: Start a new game
- **Step**: Take an action, get reward

**Key components:**
```python
class KuhhandelEnv(gym.Env):
    def __init__(self, num_players: int = 3):
        self.action_space = spaces.Discrete(7)  # 7 possible actions
        self.observation_space = None  # TODO: Define what agent sees
        
    def reset(self):
        """Start a new game"""
        self.game = Game(num_players=self.num_players).setup()
        # Create agents: 1 RL agent + RandomAgents as opponents
        
    def _decode_action(self, action_int: int, game: Game) -> GameAction:
        """Convert number (0-6) to actual game action"""
        # TODO: Implement mapping
```

**Current status:** This is still being developed! The structure exists but needs:
- Observation space definition (what the AI sees)
- Action decoding (converting numbers to actions)
- Reward function (how to score the AI's performance)

---

### 🧠 rl_agent.py - The RL Agent

**What it does:** Acts as a bridge between the RL model and the game controller.

**The challenge:** 
- The game expects actions like `Actions.bid(100)`
- The RL algorithm outputs numbers like `3`

**The solution:** The `RLAgent` converts the RL model's number output into proper `GameAction` objects.

```python
class RLAgent(Agent):
    def get_action(self, game: Game, valid_actions: List[ActionType]) -> GameAction:
        # The RL model already chose a number (stored in last_action_int)
        # Now convert it to a GameAction
        return self.env._decode_action(self.last_action_int, game)
```

---

## 🔄 How Everything Works Together

Here's the complete flow when running an RL training session:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. RL ALGORITHM (PPO from Stable-Baselines3)          │
│     - Learns from experience                           │
│     - Outputs: action number (0-6)                     │
│                                                         │
└─────────────┬───────────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  2. GYMNASIUM ENV (rl/env.py)                          │
│     - Receives action number                           │
│     - Translates to game world                         │
│     - Returns: observation, reward, done               │
│                                                         │
└─────────────┬───────────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  3. RL AGENT (rl/rl_agent.py)                          │
│     - Converts action number → GameAction              │
│     - Implements Agent interface                       │
│                                                         │
└─────────────┬───────────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  4. GAME CONTROLLER (gameengine/controller.py)         │
│     - Asks agent for action                            │
│     - Manages game flow                                │
│     - Coordinates agents and game                      │
│                                                         │
└─────────────┬───────────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  5. GAME (gameengine/game.py)                          │
│     - Executes action according to rules               │
│     - Updates game state                               │
│     - Returns results                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Step-by-Step Example:

Let's say the RL agent wants to bid in an auction:

1. **RL Algorithm decides:** "I'll take action 2" (outputs integer `2`)

2. **Gymnasium Env receives:** `action_int = 2`
   - Stores it in `RLAgent.last_action_int`

3. **Game Controller calls:** `agent.get_action(game, valid_actions)`

4. **RL Agent translates:** 
   - Looks at `valid_actions` and `game` state
   - Decodes `2` → `Actions.bid(amount=100)`

5. **Game executes:** 
   - Checks if bid is valid
   - Processes the bid
   - Updates auction state

6. **Controller continues:** Moves to next step in the game

7. **Env observes results:**
   - Game state changed
   - Calculates reward
   - Returns to RL algorithm

8. **RL Algorithm learns:** Updates its strategy based on the outcome

---

## 🚀 Getting Started - Running the Code

### Installation

```bash
# 1. Install Poetry (package manager for Python)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Clone the repository
git clone https://github.com/justingebert/kuhhandel-ml.git
cd kuhhandel-ml

# 3. Install dependencies
poetry install
```

### Running a Demo Game

The easiest way to see the game in action:

```bash
# Run a demo game with random agents
poetry run python tests/demo_game.py

# Run with specific settings
poetry run python tests/demo_game.py --seed 42 --players 4
```

This will show you:
- The game state after each turn
- What actions are taken
- Final scores

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=gameengine
```

---

## 🎓 Key Concepts to Understand

### 1. Object-Oriented Programming (OOP)

This project uses **classes** heavily. Think of a class as a blueprint:

```python
# Blueprint for a player
class Player:
    def __init__(self, player_id, name):
        self.player_id = player_id  # Attributes (properties)
        self.name = name
        self.money = []
    
    def add_money(self, cards):  # Methods (actions)
        self.money.extend(cards)

# Create an actual player (instance)
alice = Player(0, "Alice")
alice.add_money([card1, card2])
```

**Key terms:**
- **Class**: The blueprint (e.g., `Player`)
- **Instance**: An actual object created from the blueprint (e.g., `alice`)
- **Attributes**: Data stored in the object (e.g., `self.name`)
- **Methods**: Functions that belong to the object (e.g., `add_money()`)

### 2. Inheritance

Some classes **inherit** from other classes:

```python
class Agent(ABC):  # Parent class
    def get_action(self, game, valid_actions):
        pass

class RandomAgent(Agent):  # Child class inherits from Agent
    def get_action(self, game, valid_actions):
        # Implementation here
        return random.choice(valid_actions)
```

This means `RandomAgent` **is a type of** `Agent` and must implement `get_action()`.

### 3. Enums (Enumerations)

Enums are a way to define a set of named constants:

```python
class GamePhase(Enum):
    SETUP = "setup"
    PLAYER_TURN = "player_turn"
    AUCTION = "auction"

# Usage
current_phase = GamePhase.AUCTION
if current_phase == GamePhase.AUCTION:
    print("We're in an auction!")
```

**Why use enums?** They prevent typos and make code clearer than using strings.

### 4. Type Hints

You'll see type hints throughout the code:

```python
def add_money(self, cards: List[MoneyCard]) -> None:
    #                    ^^^^^^^^^^^^^^^^    ^^^^
    #                    input type          return type
```

These help:
- Document what types are expected
- Catch errors with tools like mypy
- Make code easier to understand

### 5. Dataclasses

Dataclasses are a shortcut for creating simple classes:

```python
@dataclass(frozen=True)
class BidAction:
    amount: int
    type: ActionType = field(default=ActionType.BID)

# Usage
action = BidAction(amount=100)
print(action.amount)  # 100
```

**Benefits:**
- Less boilerplate code
- Automatic `__init__`, `__repr__`, etc.
- `frozen=True` makes it immutable (can't change after creation)

### 6. Abstract Base Classes (ABC)

An ABC is a class that can't be instantiated directly - it's just a template:

```python
class Agent(ABC):
    @abstractmethod
    def get_action(self, game, valid_actions):
        pass  # Must be implemented by subclasses

# You can't do this:
agent = Agent()  # ERROR!

# But you can do this:
class MyAgent(Agent):
    def get_action(self, game, valid_actions):
        return some_action

my_agent = MyAgent()  # OK!
```

### 7. List Comprehensions

You'll see compact ways to create lists:

```python
# Traditional way
animals = []
for card in self.animals:
    if card.animal_type == AnimalType.COW:
        animals.append(card)

# List comprehension (same thing, one line)
animals = [card for card in self.animals if card.animal_type == AnimalType.COW]
```

---

## 🎯 Next Steps for Learning

### Beginner Path

1. **Read the demo game** ([tests/demo_game.py](tests/demo_game.py))
   - See how `RandomAgent` makes decisions
   - Understand the game loop

2. **Run the demo** and watch the output
   - See how game state changes
   - Observe the different phases

3. **Read Player.py** - it's simple and well-documented

4. **Try modifying RandomAgent**
   - Make it always bid low
   - Make it prefer certain animals
   - See how scores change!

### Intermediate Path

1. **Understand the Game class**
   - Read method by method
   - Focus on one phase at a time (e.g., auction)

2. **Study the Controller**
   - See how it orchestrates the game
   - Understand the interaction between Game and Agents

3. **Look at the RL layer**
   - Understand what Gymnasium does
   - Think about how to define observations

### Advanced Path

1. **Implement the observation space**
   - What should the AI see?
   - How to represent it as numbers?

2. **Define action decoding**
   - Map numbers to game actions
   - Handle invalid actions

3. **Create a reward function**
   - When should the AI get positive rewards?
   - How to encourage good strategy?

4. **Train your first RL agent!**
   - Use Stable-Baselines3's PPO
   - Watch it learn over time

---

## 🤝 Contributing and Questions

- **Ask questions:** There are no stupid questions! 
- **Experiment:** Try changing things and see what happens
- **Read error messages:** They're helpful, not scary
- **Use print statements:** Add `print()` to understand what's happening
- **Use a debugger:** Set breakpoints and step through code

### Useful Commands

```bash
# Run Python interactively
poetry run python

# Then you can experiment:
>>> from gameengine import Game
>>> game = Game(num_players=3)
>>> game.setup()
>>> print(game.players[0])
```

---

## 📚 Resources for Learning

### Python Basics
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/) - Excellent tutorials

### Reinforcement Learning
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - Great intro to RL

### Python Game Development
- Understanding game loops
- State management
- Turn-based game design

---

## 📝 Summary

**The Game Engine:**
- `game.py` = the rules and logic
- `controller.py` = runs the game
- `agent.py` = interface for any player
- `Player.py`, `Animal.py`, `Money.py` = game components

**The RL Layer:**
- `env.py` = wraps game for RL algorithms
- `rl_agent.py` = bridges RL and game

**How they connect:**
- RL Algorithm → Gymnasium Env → RL Agent → Controller → Game
- Game state flows back through the same chain

**Your mission:**
1. Understand the game rules
2. See how the code implements those rules
3. Learn how RL integration works
4. Eventually: make the AI smarter!

---

Good luck, and welcome to the team! 🎉

*Questions? Don't hesitate to ask - we're here to help!*

# 🎲 Kuhhandel RL Game Project

A reinforcement learning environment and training pipeline for the card game *Kuhhandel*, built with Python, Gymnasium, and Stable Baselines 3.

## 👥 Authors

- Justin Gebert
- Florian Hering  
- Nepomuk Aurich

---

## 📖 About the Game

*Kuhhandel* is a strategic card game involving auctions and trading. Players compete to collect complete sets of animal cards.

📜 **[Game Rules](https://www.spiele4us.de/wp-content/uploads/2022/10/kuhhandel-ravensburger-n003700650.pdf)**

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Game Engine   │────▶│  RL Environment │────▶│   PPO Agent     │
│  (gameengine/)  │     │   (rl/env.py)   │     │ (MaskablePPO)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **Game Engine**: Core game logic, rules, and state management
- **RL Environment**: Gymnasium-compatible wrapper with observation/action spaces
- **PPO Agent**: Maskable PPO for action masking with invalid action handling

---

## 🤖 RL Approach

- **Algorithm**: Maskable PPO (sb3-contrib) with action masking for invalid moves
- **Training**: Self-play with opponent pool (random + historical models)
- **Reward Shaping**: Configurable reward functions (see `rl/rewardconfigs/`)

---

## 📁 Project Structure

```
kuhhandel-ml/
├── gameengine/          # Core game implementation
│   ├── game.py          # Main game logic and state machine
│   ├── Animal.py        # Animal card definitions
│   ├── Money.py         # Money card and deck logic
│   ├── Player.py        # Player state management
│   ├── actions.py       # Action types and factory
│   ├── agent.py         # Abstract agent interface
│   └── controller.py    # Game flow controller
├── rl/                  # Reinforcement learning components
│   ├── env.py           # Gymnasium environment wrapper
│   ├── agents/          # Agent implementations (random, model, user)
│   ├── train/           # Training scripts and configs
│   ├── models/          # Trained model checkpoints
│   └── rewardconfigs/   # Reward shaping configurations
├── tests/               # Pytest test suite
├── gui_game.py          # PyQt6 GUI for playing against AI
└── plots/               # Training visualization scripts
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- Poetry

1. **Clone and install dependencies**:
```bash
git clone https://github.com/justingebert/kuhhandel-ml.git
cd kuhhandel-ml
poetry install --with dev,ml
```

---

## 🚀 Usage

### Play the Game (GUI)

Play against AI opponents using the graphical interface:

```bash
poetry run python gui_game.py
```

### Train a Model

Run self-play training:

```bash
poetry run python rl/train/train_selfplay.py
```

**Options:**
- `--itp`: Use ITP server configuration
- `--preset`: Hyperparameter preset (`default`, `low_lr`, `high_lr`)

### Evaluate Model Performance

Compare model win rates:

```bash
# Model vs Random opponents
poetry run python rl/evaluate_winrate.py --main rl/models/gen150win_only.zip --n 100

# Model vs Model
poetry run python rl/evaluate_winrate.py --main rl/models/gen150win_only.zip --opp rl/models/gen150oldRew.zip --n 100
```

### Run Tests

```bash
poetry run pytest tests/ -v
```

With coverage:
```bash
poetry run pytest tests/ --cov=gameengine --cov-report=html
```

---

## 📊 Results

<!-- TODO: Add results from slides -->
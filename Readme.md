# 🎲 Kuhhandel (RL Game Project)

A complete Python implementation of the card game *Kuhhandel* (You're Bluffing) - built to explore reinforcement learning and game simulation.

## 👥 Authors

- Justin Gebert
- Florian Hering  
- Nepomuk Aurich

---

## ⚙️ Setup

1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
2. Clone and install
```bash
git clone https://github.com/justingebert/kuhhandel-ml.git
cd kuhhandel-ml
poetry install
```


### notes:
how will observations be managed? 


TODO



- define public observations
    - action space 
    - money
        - gegnerisches geld soweit bekannt aus versteigerung (nimmt an, dass man möglichst günstig bezahlt)
        - anzahl geldkarten gegner
    - animals
        - tierkarten Spieler
        - tierkarten verbleibend
    - phase
        - später: welche spieler haben gehandelt? (wer hat gewonnen?)
            1-2 (vorne gewinnt) = Anzahl Handel / gesamthandel anzahl 2/4
            2-1 = 
            2-3 = 2/4
            3-2 = 2/4 * faktor für die gehandelten tiere
            1-3 
            3-1
        - welche runde?
        - wer ist dran?

- private observations:
    - eigenes geld


- reward functions
- algorithms for imperfect information games

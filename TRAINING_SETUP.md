# Kuhhandel RL Training Guide

## 1. Aktueller Stand (Single-Agent vs. Random)

Das aktuelle Training ist darauf ausgelegt, dass **ein** Reinforcement Learning (RL) Agent gegen fest programmierte Zufalls-Bots (`RandomAgent`) spielt.

### Dateien & Ablauf
*   **`rl/train.py`**: Der Einstiegspunkt.
    *   Initialisiert die Umgebung `KuhhandelEnv`.
    *   Nutzt `sb3_contrib.MaskablePPO` als Algorithmus.
    *   Trainiert für 50.000 Timesteps.
*   **`rl/env.py` (`KuhhandelEnv`)**: Die Gym-Umgebung.
    *   **`reset()`**: Erstellt immer *einen* `RLAgent` (Spieler 0) und füllt die restlichen Plätze mit `RandomAgent`-Instanzen.
    *   **`step()`**: Führt die Aktion des RL-Agenten aus und simuliert dann automatisch ("fast-forward") die Züge der Gegner, bis der RL-Agent wieder am Zug ist.
    *   **Observation**: Die Methode `_get_observation()` ist derzeit hardcoded, um die Sicht von Spieler 0 (dem RL-Agenten) zurückzugeben.

### Limitierungen
Der Agent lernt nur, die Schwächen von zufälligen Spielern auszunutzen. Er entwickelt keine fortgeschrittenen Strategien (wie Bluffen oder gezieltes Bieten), da die Gegner darauf nicht reagieren.

---

## 2. Anleitung: Multi-Agent Training (Self-Play) einrichten

Um starke Agenten zu trainieren, sollten diese gegeneinander spielen ("Self-Play"). Dabei spielt die aktuelle Version des Agenten gegen frühere Versionen seiner selbst.

### Schritt 1: Environment Refactoring (`rl/env.py`)
Die Umgebung muss flexibler werden, um Observationen für *jeden* Spieler generieren zu können, nicht nur für Spieler 0.

*   **Aufgabe**: Die Methoden `_get_observation()` und `get_action_mask()` müssen einen Parameter `player_id` akzeptieren.
*   **Ziel**: Der RL-Algorithmus kann nach der Situation für Spieler X fragen, nicht nur für den "Haupt-Agenten".

### Schritt 2: ModelAgent Implementierung
Wir benötigen eine Agenten-Klasse, die genau wie `RandomAgent` funktioniert, aber Entscheidungen basierend auf einem gespeicherten Modell trifft.

*   **Neue Klasse `ModelAgent`**:
    *   Lädt ein gespeichertes PPO-Modell.
    *   In `get_action()`:
        1.  Holt Observation für sich selbst (via neuer Env-Methode).
        2.  Fragt das Modell (`model.predict(obs)`).
        3.  Wandelt die Ausgabe wieder in eine `GameAction` um.

### Schritt 3: Gegner-Injektion
Die `KuhhandelEnv` muss erlauben, die Gegner von außen zu bestimmen, statt immer `RandomAgent` zu nutzen.

*   Anpassung `__init__` oder `reset`: Akzeptiere eine Liste von Agenten oder eine Funktion `opponent_generator`.
*   Damit können wir im Training dynamisch Agenten austauschen.

### Schritt 4: Self-Play Trainings-Loop
Statt dem einfachen `model.learn()` in `rl/train.py` benötigen wir einen iterativen Loop:

1.  **Initialisierung**: Starte mit einem Random-Modell.
2.  **Loop (Generationen)**:
    *   Lade Gegner (z.B. 50% Random, 50% das beste Model der letzten Generation).
    *   Trainiere den aktuellen Agenten für X Timesteps gegen diese Gegner.
    *   Speichere das neue Modell als "Generation N".
    *   Setze das neue Modell ab sofort als einen der möglichen Gegner ein.

---

## Nächste Schritte zur Umsetzung

1.  **Refactoring**: `rl/env.py` anpassen, damit `_get_observation(player_id)` unterstützt wird.
2.  **Klasse erstellen**: `rl/model_agent.py` anlegen.
3.  **Skript erstellen**: `rl/train_selfplay.py` für den neuen Loop.

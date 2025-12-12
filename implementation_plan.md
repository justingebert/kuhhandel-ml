# Optimization Implementation Plan

## Goal
Improve training performance by:
1.  **Caching Models**: Prevent reloading models from disk every episode.
2.  **Vectorization**: Use parallel environments (`SubprocVecEnv`) to utilize multi-core CPUs.

## User Review Required
> [!IMPORTANT]
> Vectorized environments require `if __name__ == "__main__":` guards and careful handling of global state. The current script seems to have this, but I will double check.

## Proposed Changes

### 1. `rl/model_agent.py` & `rl/train_selfplay.py`
*   **Model Caching**: Introduce a dictionary `MODEL_CACHE = {}` in `train_selfplay.py`.
*   Update `create_opponents_with_env` to check the cache before loading `MaskablePPO.load()`.
    *   If cached, use `ModelAgent(..., model=cached_model)`.
    *   Need to refactor `ModelAgent` to accept an existing model instance instead of just a path.

### 2. `rl/train_selfplay.py` - Vectorization
*   Switch from `Monitor(KuhhandelEnv(...))` to `make_vec_env`.
*   Use `SubprocVecEnv` for parallel execution.
*   Update the `generator` logic because `env` cannot be easily passed into subprocesses if it's not picklable or if it refers to the main process env.
    *   **Challenge**: The `opponent_generator` depends on `env` reference to get observations for specific players.
    *   **Solution**: In `SubprocVecEnv`, each process has its own `Env` instance. The `opponent_generator` should rely on the `env` instance *inside* that process.
    *   The `env` passed to `ModelAgent` must be the local process env.
    *   I need to ensure `KuhhandelEnv` passes `self` to the generator correctly.

#### Refactoring `rl/env.py`
*   Update `reset()` to pass `self` to `opponent_generator`:
    `others = self.opponent_generator(self.num_players - 1, self)`
*   This removes the need for the closure hack in `train_selfplay.py`.

### Files to Modify

#### [MODIFY] [env.py](file:///c:/Users/jnnau/Desktop/Kuhandel%20Engine/kuhhandel-ml/rl/env.py)
*   Update `reset()` to call `self.opponent_generator(n, self)`.

#### [MODIFY] [model_agent.py](file:///c:/Users/jnnau/Desktop/Kuhandel%20Engine/kuhhandel-ml/rl/model_agent.py)
*   Update `__init__` to accept `model_instance` directly, optional `model_path`.

#### [MODIFY] [train_selfplay.py](file:///c:/Users/jnnau/Desktop/Kuhandel%20Engine/kuhhandel-ml/rl/train_selfplay.py)
*   Implement `MODEL_CACHE`.
*   Implement `make_env` function for Vectorization.
*   Use `SubprocVecEnv(n_envs=4 or 8)`.

## Verification Plan

### Automated Tests
Run `python -m rl.train_selfplay` and check:
1.  **FPS**: Should increase significantly (monitor console output).
2.  **CPU Usage**: Should rise (multiple python processes).
3.  **No Errors**: Generation loop completes 1-2 generations.

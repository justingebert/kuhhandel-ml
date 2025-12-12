# Optimization Implementation Plan

## Goal
Improve training performance by maximizing throughput to feed the GPU efficiently.
*   **Problem**: Current setup runs 1 game at a time. The GPU sits idle waiting for the CPU to finish game logic.
*   **Solution**: Run many games in parallel (Vectorization) to generate large batches of data. This keeps the GPU busy.

## User Review Required
> [!IMPORTANT]
> This plan implements **CPU Vectorization** (`SubprocVecEnv`). This is the standard way to optimize Gym environments for GPU training. It allows us to step 16+ environments at once, creating large batches for the GPU.

## Proposed Changes

### 1. `rl/train_selfplay.py` - Parallelization
*   **Vectorization**: Switch to `SubprocVecEnv` to run multiple game instances on separate CPU cores.
    *   Target: Use `multiprocessing.cpu_count()` or a fixed number (e.g., 8-16) environments.
*   **Batching**: The PPO model will now receive a batch of observations (Shape: `[n_envs, obs_size]`) and output a batch of actions. This is where the GPU provides speedup.
*   **Model Caching**: Implement a `MODEL_CACHE` to avoid disk I/O bottlenecks during environment resets in subprocesses.

### 2. `rl/env.py` - Compatibility
*   **Generator Update**: The `opponent_generator` needs to handle being called inside a subprocess.
    *   Update `reset()` to pass `self` (the local process environment instance/reference) to the generator.
    *   This ensures each parallel environment manages its own opponents correctly.

### 3. `rl/model_agent.py` - Efficiency
*   **Inference**: Opponent bots currently use `model.predict(obs)`.
    *   For now, each opponent in each subprocess calls its own model.
    *   *Optimization*: Ensure we don't reload weights from disk. Pass loaded model objects or share memory if possible (SB3 models are PyTorch, sharing is tricky across processes, but caching per process is enough).

## Verification Plan
1.  **FPS Monitoring**: Check if `fps` in logs increases (Target: >500-1000 fps vs current low fps).
2.  **GPU Utilization**: Check if `nvidia-smi` (or Task Manager) shows spikes in usage during `learn()`.

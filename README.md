---
title: Trading Execution Environment Server
emoji: chart
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Trading Execution Environment

An OpenEnv-compatible reinforcement learning environment for optimal trade execution with simulated market impact and slippage-aware rewards.

## Quick Start

The easiest way to use the environment is through `TradingExecutionEnv`.

```python
from trading_execution_env import TradingExecutionAction, TradingExecutionEnv

with TradingExecutionEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print("Remaining:", result.observation.remaining_quantity)

    # Execute 500 shares as a market order
    result = env.step(TradingExecutionAction(quantity=500.0, order_type="market"))
    print("VWAP:", result.observation.vwap)
    print("Slippage:", result.observation.slippage)
    print("Reward:", result.reward)
```

## Build and Run

Build Docker image:

```bash
docker build -t trading_execution_env-env:latest -f server/Dockerfile .
```

Run server locally (without Docker):

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Or via project script:

```bash
uv run --project . server
```

## Environment Interface

### Action

`TradingExecutionAction`

- `quantity` (`float`): Number of shares to trade this step.
- `order_type` (`str`): `"market"` or `"limit"`.
- `limit_price` (`float | None`): Optional price used for limit orders.

### Observation

`TradingExecutionObservation`

- `bid` (`float`): Current best bid.
- `ask` (`float`): Current best ask.
- `mid_price` (`float`): Mid-market price.
- `volume` (`float`): Simulated market volume.
- `remaining_quantity` (`float`): Shares left to execute.
- `time_remaining` (`int`): Steps left in the episode.
- `vwap` (`float`): Volume weighted average fill price.
- `slippage` (`float`): Cumulative slippage.
- `filled` (`float`): Cumulative executed quantity.
- `done` (`bool`): Episode completion flag.
- `task_id` (`str`): Active task identifier.
- `reward` (`float`): Reward emitted by environment logic.
- `metadata` (`dict`): Extra diagnostic values.

## Available Tasks

Configured tasks (see `openenv.yaml`):

- `simple-fill`: 10,000 shares, 20 steps, low volatility.
- `adaptive-execution`: 10,000 shares, 30 steps, higher volatility.
- `multi-asset`: 30,000 shares, 40 steps, medium volatility.

## Reward Design

Reward combines:

- Fill progress (higher is better).
- Slippage penalty (lower is better).
- Completion bonus for fully filling by episode end.

The final reward is clipped to the range `[0.0, 1.0]`.

## Deploy to Hugging Face Spaces

From this environment directory:

```bash
openenv push
```

Common options:

```bash
openenv push --repo-id my-org/my-env
openenv push --private
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest
```

After deployment:

- Web UI: `/web`
- API docs: `/docs`
- Health: `/health`
- WebSocket endpoint: `/ws`

## Project Structure

```text
trading_execution_env/
+-- __init__.py
+-- README.md
+-- client.py
+-- models.py
+-- openenv.yaml
+-- pyproject.toml
+-- uv.lock
+-- server/
    +-- __init__.py
    +-- app.py
    +-- trading_execution_env_environment.py
    +-- requirements.txt
    +-- Dockerfile
```

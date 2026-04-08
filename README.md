---
title: Trading Execution Env
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
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

**Price & Volume:**
- `bid` (`float`): Current best bid.
- `ask` (`float`): Current best ask.
- `mid_price` (`float`): Mid-market price.
- `price_momentum` (`float`): Price change over last 3 steps (3-step return).
- `volatility` (`float`): Recent price volatility (std of last 3 returns).
- `volume` (`float`): Current market volume (U-shaped intraday curve).

**Order Status:**
- `remaining_quantity` (`float`): Shares left to execute.
- `filled` (`float`): Cumulative executed quantity.
- `fill_rate` (`float`): Current fill rate as a fraction of target.
- `time_remaining` (`int`): Steps left in the episode.

**Execution Metrics:**
- `vwap` (`float`): Volume weighted average fill price.
- `slippage` (`float`): Cumulative slippage (cost vs mid-market).
- `done` (`bool`): Episode completion flag.

**Metadata:**
- `task_id` (`str`): Active task identifier.
- `reward` (`float`): Reward emitted by environment logic.
- `metadata` (`dict`): Extra diagnostic values (total_reward, step count).

## Available Tasks

Configured tasks (see `openenv.yaml`):

- **simple-fill**: 10,000 shares in 20 steps, low volatility (0.01). Baseline easy task for testing.
- **adaptive-execution**: 10,000 shares in 30 steps, high volatility (0.03). Tests adaptability under uncertain conditions.
- **multi-asset**: 30,000 shares in 40 steps, medium volatility (0.02). Large order with moderate time pressure.

### Market Realism

The market simulator includes:

- **Geometric Brownian Motion** for price evolution.
- **U-shaped volume curve** that mimics real intraday patterns (high volume at open & close).
- **Market impact model** that scales with order size relative to available volume.
- **Price momentum & volatility tracking** to support adaptive execution.

## Reward Design

Reward is carefully shaped to encourage complete order execution while penalizing slippage:

**Base Reward:**
- Fill ratio: `filled / total_qty` (fraction of order completed).
- Slippage penalty: `-10 × (slippage / (total_qty × initial_price))`.

**Completion Bonus:**
- **+0.3** if full order is filled by episode end (big bonus for success).
- **-0.5 × unfilled_ratio** if order is incomplete at timeout (heavy penalty for failure).

The final reward is clipped to `[0.0, 1.0]`. This design strongly incentivizes agents to:
1. Fill orders completely (not just partially).
2. Minimize slippage and market impact.
3. Manage time urgency intelligently.

## Execution Policy

The baseline execution policy is **LLM-driven** with urgency-aware prompting:

- **Normal execution** (>5 steps remaining): Even pacing via TWAP baseline.
- **Warning** (≤5 steps): Aggregate urgency message; agent increases pace.
- **Urgent** (≤3 steps): Explicit instruction to trade remaining shares aggressively.

The LLM sees the full market state plus indicators:
- Current bid/ask, momentum, and volatility
- Days-to-fill metrics (remaining qty, fill_rate, time_remaining)
- Historical metrics (VWAP, cumulative slippage)

The agent also has a deterministic fallback: if JSON parsing fails, it executes TWAP with a 1.5x multiplier if time is running out.

## Running Inference

To run the inference script and test the policy:

```bash
python inference.py
```

This runs all three tasks sequentially and outputs JSON logs for each step and final result. Each task logs:
- `[START]`: Task initialization.
- `[STEP]`: Per-step action, reward, fill status, slippage.
- `[END]`: Final score, total steps, total filled, total slippage.

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trading Execution Environment Implementation.

An RL environment where an agent learns to execute large orders
with minimal slippage using a simulated market.
"""

import numpy as np
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TradingExecutionAction, TradingExecutionObservation
except ImportError:
    from models import TradingExecutionAction, TradingExecutionObservation


TASKS = {
    "simple-fill": {
        "total_qty": 10000,
        "total_steps": 20,
        "volatility": 0.01,
        "description": "Fill a fixed order with minimal slippage"
    },
    "adaptive-execution": {
        "total_qty": 10000,
        "total_steps": 30,
        "volatility": 0.03,   # more volatile market
        "description": "Execute in volatile market conditions"
    },
    "multi-asset": {
        "total_qty": 30000,
        "total_steps": 40,
        "volatility": 0.02,
        "description": "Execute a large order across many steps"
    },
}


class MarketSimulator:
    """Simulates a simple market using Geometric Brownian Motion."""

    def __init__(self, seed=None, initial_price=100.0, volatility=0.02):
        self._rng = np.random.default_rng(seed)
        self.initial_price = initial_price
        self.price = initial_price
        self.volatility = volatility
        self._vwap_sum = 0.0
        self._vwap_vol = 0.0
        self.tick()  # initialize bid/ask/volume

    def tick(self):
        """Advance market by one step."""
        self.price *= np.exp(self._rng.normal(0, self.volatility))
        spread = self.price * 0.001
        self.bid = round(self.price - spread / 2, 4)
        self.ask = round(self.price + spread / 2, 4)
        self.mid = round(self.price, 4)
        self.volume = float(self._rng.lognormal(mean=8, sigma=0.5))

    def execute(self, qty: float, order_type: str = "market") -> float:
        """Execute an order and return fill price."""
        # Market impact: larger orders move price more
        impact = (qty / max(self.volume, 1)) * self.price * 0.1
        fill_price = self.ask + impact if order_type == "market" else self.ask
        fill_price = round(fill_price, 4)
        # Track VWAP
        self._vwap_sum += fill_price * qty
        self._vwap_vol += qty
        return fill_price

    @property
    def vwap(self) -> float:
        if self._vwap_vol == 0:
            return self.price
        return round(self._vwap_sum / self._vwap_vol, 4)


class TradingExecutionEnvironment(Environment):
    """
    Trading Execution RL Environment.

    The agent must execute a large order (e.g. 10,000 shares) over
    multiple steps while minimizing slippage against a simulated market.

    Tasks:
        - simple-fill: Fixed order, stable market (easy)
        - adaptive-execution: Fixed order, volatile market (medium)
        - multi-asset: Large order, volatile market (hard)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "simple-fill"):
        self.task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.market = None
        self.total_qty = 0
        self.total_steps = 0
        self.filled = 0.0
        self.slippage = 0.0
        self.total_reward = 0.0

    def reset(self) -> TradingExecutionObservation:
        """Reset the environment and start a new episode."""
        task = TASKS[self.task_id]
        self.total_qty = task["total_qty"]
        self.total_steps = task["total_steps"]
        self.filled = 0.0
        self.slippage = 0.0
        self.total_reward = 0.0
        self.market = MarketSimulator(
            seed=None,
            initial_price=100.0,
            volatility=task["volatility"]
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return TradingExecutionObservation(
            bid=self.market.bid,
            ask=self.market.ask,
            mid_price=self.market.mid,
            volume=self.market.volume,
            remaining_quantity=self.total_qty,
            time_remaining=self.total_steps,
            vwap=self.market.vwap,
            slippage=0.0,
            filled=0.0,
            done=False,
            task_id=self.task_id,
            reward=0.0,
        )

    def step(self, action: TradingExecutionAction) -> TradingExecutionObservation:
        """Execute one trading step."""
        self._state.step_count += 1

        # Clamp quantity to what's remaining
        qty = min(max(action.quantity, 0.0), self.total_qty - self.filled)

        # Execute in market
        fill_price = self.market.execute(qty, action.order_type)

        # Track slippage (cost vs mid price)
        self.slippage += (fill_price - self.market.mid) * qty
        self.filled += qty

        # Advance market
        self.market.tick()

        done = (self.filled >= self.total_qty) or \
               (self._state.step_count >= self.total_steps)

        reward = self._calculate_reward(done)
        self.total_reward += reward

        return TradingExecutionObservation(
            bid=self.market.bid,
            ask=self.market.ask,
            mid_price=self.market.mid,
            volume=self.market.volume,
            remaining_quantity=round(self.total_qty - self.filled, 2),
            time_remaining=self.total_steps - self._state.step_count,
            vwap=self.market.vwap,
            slippage=round(self.slippage, 4),
            filled=round(self.filled, 2),
            done=done,
            task_id=self.task_id,
            reward=round(reward, 4),
            metadata={
                "total_reward": round(self.total_reward, 4),
                "step": self._state.step_count,
            }
        )

    def _calculate_reward(self, done: bool) -> float:
        """
        Reward function:
        - Fill ratio: how much of the order is filled (0.0 - 1.0)
        - Slippage penalty: penalize high slippage
        - Completion bonus: bonus if fully filled
        """
        fill_ratio = self.filled / self.total_qty
        slippage_penalty = self.slippage / (self.total_qty * self.market.initial_price)
        reward = fill_ratio - abs(slippage_penalty) * 10

        # Bonus for completing the order
        if done and self.filled >= self.total_qty:
            reward += 0.2

        return round(max(0.0, min(1.0, reward)), 4)

    @property
    def state(self) -> State:
        """Get the current episode state."""
        return self._state
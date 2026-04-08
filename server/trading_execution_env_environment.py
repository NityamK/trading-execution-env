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
        "volatility": 0.03,
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

    def __init__(self, seed=None, initial_price=100.0, volatility=0.02, total_steps=40):
        self._rng = np.random.default_rng(seed)
        self.initial_price = initial_price
        self.price = initial_price
        self.volatility = volatility
        self.total_steps = max(total_steps, 1)
        self.step_index = 0
        self._vwap_sum = 0.0
        self._vwap_vol = 0.0
        self._price_history = [self.price]
        self.tick()  # initialize bid/ask/volume

    def get_volume_curve(self, step, total_steps):
        """U-shaped intraday pattern with higher volume near the open and close."""
        t = step / max(total_steps - 1, 1)
        return 1.0 + 2.0 * (t - 0.5) ** 2

    def tick(self):
        """Advance market by one step."""
        self.price *= np.exp(self._rng.normal(0, self.volatility))
        spread = self.price * 0.001
        self.bid = round(self.price - spread / 2, 4)
        self.ask = round(self.price + spread / 2, 4)
        self.mid = round(self.price, 4)
        base_volume = float(self._rng.lognormal(mean=8, sigma=0.5))
        self.volume = base_volume * self.get_volume_curve(self.step_index, self.total_steps)
        self._price_history.append(self.price)
        self.step_index += 1

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
    def price_momentum(self) -> float:
        if len(self._price_history) < 4:
            return 0.0
        baseline = self._price_history[-4]
        if baseline == 0:
            return 0.0
        return round((self.price - baseline) / baseline, 6)

    @property
    def recent_volatility(self) -> float:
        if len(self._price_history) < 3:
            return 0.0
        recent_prices = np.array(self._price_history[-4:])
        returns = np.diff(recent_prices) / np.maximum(recent_prices[:-1], 1e-9)
        if len(returns) == 0:
            return 0.0
        return round(float(np.std(returns)), 6)

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
        self.fill_rate = 0.0
        self.slippage = 0.0
        self.total_reward = 0.0
        # Note: task selection is controlled via reset(task_id=...).

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> TradingExecutionObservation:
        """Reset the environment and start a new episode.

        Args:
            seed: Unused but kept for compatibility with the base interface.
            episode_id: Optional external episode identifier.
            task_id: Optional task selector ("simple-fill", "adaptive-execution", "multi-asset").
        """
        # Update task if provided and valid; otherwise keep current.
        if task_id is not None and task_id in TASKS:
            self.task_id = task_id

        task = TASKS[self.task_id]
        self.total_qty = task["total_qty"]
        self.total_steps = task["total_steps"]
        self.filled = 0.0
        self.slippage = 0.0
        self.total_reward = 0.0
        self.market = MarketSimulator(
            seed=seed,
            initial_price=100.0,
            volatility=task["volatility"],
            total_steps=self.total_steps,
        )
        # If caller provided an explicit episode_id, honor it; otherwise generate one.
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self.fill_rate = 0.0

        return TradingExecutionObservation(
            bid=self.market.bid,
            ask=self.market.ask,
            mid_price=self.market.mid,
            price_momentum=self.market.price_momentum,
            volatility=self.market.recent_volatility,
            volume=self.market.volume,
            remaining_quantity=self.total_qty,
            fill_rate=self.fill_rate,
            time_remaining=self.total_steps,
            vwap=self.market.vwap,
            slippage=0.0,
            filled=0.0,
            done=False,
            task_id=self.task_id,
            reward=0.0,
        )

    def step(self, action: TradingExecutionAction) -> TradingExecutionObservation:
        self._state.step_count += 1

        # Clamp quantity to what's remaining
        qty = min(max(action.quantity, 0.0), self.total_qty - self.filled)

        # Execute in market
        fill_price = self.market.execute(qty, action.order_type)

        self.slippage += (fill_price - self.market.mid) * qty
        self.filled += qty
        self.fill_rate = self.filled / max(self.total_qty, 1)
        self.market.tick()

        done = (self.filled >= self.total_qty) or \
            (self._state.step_count >= self.total_steps)

        # Force fill remaining on last step
        if self._state.step_count >= self.total_steps and self.filled < self.total_qty:
            remaining = self.total_qty - self.filled
            fill_price = self.market.execute(remaining, "market")
            self.slippage += (fill_price - self.market.mid) * remaining
            self.filled += remaining
            self.fill_rate = self.filled / max(self.total_qty, 1)
            done = True

        reward = self._calculate_reward(done)
        self.total_reward += reward

        return TradingExecutionObservation(
            bid=self.market.bid,
            ask=self.market.ask,
            mid_price=self.market.mid,
            price_momentum=self.market.price_momentum,
            volatility=self.market.recent_volatility,
            volume=self.market.volume,
            remaining_quantity=round(self.total_qty - self.filled, 2),
            fill_rate=round(self.fill_rate, 6),
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
        fill_ratio = self.filled / self.total_qty
        slippage_penalty = self.slippage / (self.total_qty * self.market.initial_price)
        reward = fill_ratio - abs(slippage_penalty) * 10

        # Big bonus for completing the order
        if done and self.filled >= self.total_qty:
            reward += 0.3

        # Heavy penalty for not filling when time runs out
        if done and self.filled < self.total_qty:
            unfilled_ratio = (self.total_qty - self.filled) / self.total_qty
            reward -= unfilled_ratio * 0.5

        return round(max(0.0, min(1.0, reward)), 4)

    @property
    def state(self) -> State:
        """Get the current episode state."""
        return self._state
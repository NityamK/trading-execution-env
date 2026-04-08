# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trading Execution Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TradingExecutionAction, TradingExecutionObservation
except ImportError:
    from models import TradingExecutionAction, TradingExecutionObservation


class TradingExecutionEnv(
    EnvClient[TradingExecutionAction, TradingExecutionObservation, State]
):
    """
    Client for the Trading Execution Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TradingExecutionEnv(base_url="http://localhost:8000", task_id="simple-fill") as client:
        ...     result = client.reset()
        ...     print(result.observation.remaining_quantity)
        ...
        ...     result = client.step(TradingExecutionAction(quantity=500.0, order_type="market"))
        ...     print(result.observation.vwap)
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TradingExecutionEnv.from_docker_image("trading_execution_env-env:latest", task_id="adaptive-execution")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TradingExecutionAction(quantity=500.0, order_type="market"))
        ... finally:
        ...     client.close()
    """

    def __init__(self, base_url: str = "http://localhost:8000", task_id: str = "simple-fill"):
        super().__init__(base_url=base_url)
        self.task_id = task_id

    def _step_payload(self, action: TradingExecutionAction) -> Dict:
        """
        Convert TradingExecutionAction to JSON payload for step message.

        Args:
            action: TradingExecutionAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {
            "quantity": action.quantity,
            "order_type": action.order_type,
        }
        if action.limit_price is not None:
            payload["limit_price"] = action.limit_price
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[TradingExecutionObservation]:
        """Parse server response into StepResult[TradingExecutionObservation]."""
        obs_data = payload.get("observation", {})
        observation = TradingExecutionObservation(
            bid=obs_data.get("bid", 0.0),
            ask=obs_data.get("ask", 0.0),
            mid_price=obs_data.get("mid_price", 0.0),
            price_momentum=obs_data.get("price_momentum", 0.0),
            volatility=obs_data.get("volatility", 0.0),
            volume=obs_data.get("volume", 0.0),
            remaining_quantity=obs_data.get("remaining_quantity", 0.0),
            fill_rate=obs_data.get("fill_rate", 0.0),
            time_remaining=obs_data.get("time_remaining", 0),
            vwap=obs_data.get("vwap", 0.0),
            slippage=obs_data.get("slippage", 0.0),
            filled=obs_data.get("filled", 0.0),
            done=obs_data.get("done", payload.get("done", False)),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            metadata=obs_data.get("metadata", {}),
            task_id=obs_data.get("task_id", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
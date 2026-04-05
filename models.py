# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Trading Execution Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional


class TradingExecutionAction(Action):
    """Action for the Trading Execution environment."""

    quantity: float = Field(..., description="Number of shares to trade this step")
    order_type: str = Field(default="market", description="Order type: 'market' or 'limit'")
    limit_price: Optional[float] = Field(default=None, description="Limit price (only for limit orders)")


class TradingExecutionObservation(Observation):
    """Observation from the Trading Execution environment."""

    bid: float = Field(default=0.0, description="Current best bid price")
    ask: float = Field(default=0.0, description="Current best ask price")
    mid_price: float = Field(default=0.0, description="Mid price (bid+ask)/2")
    volume: float = Field(default=0.0, description="Current market volume")
    remaining_quantity: float = Field(default=0.0, description="Shares still to be executed")
    time_remaining: int = Field(default=0, description="Steps remaining in episode")
    vwap: float = Field(default=0.0, description="Volume weighted average price so far")
    slippage: float = Field(default=0.0, description="Total slippage cost incurred so far")
    filled: float = Field(default=0.0, description="Total shares filled so far")
    done: bool = Field(default=False, description="Whether the episode is complete")
    task_id: str = Field(default="", description="Current task identifier")
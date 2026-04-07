# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference script for Trading Execution Environment.
Uses an LLM agent to execute trades via the OpenEnv interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from openai import OpenAI
from client import TradingExecutionEnv
from models import TradingExecutionAction

# ── Environment variables ──────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
SPACE_URL    = os.environ.get("SPACE_URL", "http://localhost:8000")


# print("GROQ KEY:", os.environ.get("GROQ_API_KEY", "NOT FOUND")[:15])
llm = OpenAI(
    base_url=API_BASE_URL,
    api_key=GROQ_API_KEY,
)

TASKS = ["simple-fill", "adaptive-execution", "multi-asset"]


def get_llm_action(obs, task_id: str) -> TradingExecutionAction:
    """Ask the LLM what action to take given current market observation."""

    prompt = f"""You are an expert trade execution agent.
Your goal is to fill a large order with minimal slippage.

Current market state:
- Task: {task_id}
- Bid: {obs.bid:.4f}
- Ask: {obs.ask:.4f}
- Mid price: {obs.mid_price:.4f}
- Market volume: {obs.volume:.0f}
- Shares remaining to fill: {obs.remaining_quantity:.0f}
- Steps remaining: {obs.time_remaining}
- Current VWAP: {obs.vwap:.4f}
- Slippage so far: {obs.slippage:.4f}

Decide how many shares to buy this step.
Rules:
- quantity must be between 0 and {obs.remaining_quantity:.0f}
- If time is running out, be more aggressive
- If slippage is high, trade smaller quantities
- order_type should always be "market"

Reply with ONLY a JSON object, no explanation:
{{"quantity": <number>, "order_type": "market"}}"""

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up response in case LLM adds extra text
    try:
        # Find JSON in response
        start = raw.index("{")
        end = raw.rindex("}") + 1
        action_json = json.loads(raw[start:end])
        return TradingExecutionAction(
            quantity=float(action_json["quantity"]),
            order_type=action_json.get("order_type", "market"),
        )
    except Exception:
        # Fallback: trade evenly across remaining steps
        safe_qty = obs.remaining_quantity / max(obs.time_remaining, 1)
        return TradingExecutionAction(quantity=safe_qty, order_type="market")


def run_task(task_id: str):
    """Run a single task episode and print structured logs."""

    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "space_url": SPACE_URL,
        "model": MODEL_NAME,
    }), flush=True)

    with TradingExecutionEnv(base_url=SPACE_URL).sync() as env:
        result = env.reset()
        obs = result.observation
        total_reward = 0.0
        step = 0

        while not obs.done:
            action = get_llm_action(obs, task_id)

            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            total_reward += reward
            step += 1

            print(json.dumps({
                "type": "[STEP]",
                "step": step,
                "action": {
                    "quantity": action.quantity,
                    "order_type": action.order_type,
                },
                "reward": round(reward, 4),
                "filled": obs.filled,
                "remaining": obs.remaining_quantity,
                "slippage": obs.slippage,
                "vwap": obs.vwap,
            }), flush=True)

    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "total_steps": step,
        "final_filled": obs.filled,
        "final_slippage": obs.slippage,
    }), flush=True)


def main():
    """Run all tasks sequentially."""
    print(f"Starting inference with model: {MODEL_NAME}", file=sys.stderr)
    print(f"Connecting to environment at: {SPACE_URL}", file=sys.stderr)

    for task_id in TASKS:
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Running task: {task_id}", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)
        try:
            run_task(task_id)
        except Exception as e:
            print(json.dumps({
                "type": "[END]",
                "task_id": task_id,
                "error": str(e),
                "total_reward": 0.0,
            }), flush=True)
            print(f"Task {task_id} failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
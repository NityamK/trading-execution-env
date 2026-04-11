"""
Inference Script for Trading Execution Environment
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from client import TradingExecutionEnv
from models import TradingExecutionAction

# ── Environment variables ──────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
SPACE_URL    = os.getenv("SPACE_URL") or "https://nk003-trading-execution-env.hf.space"

# ── Task configuration ─────────────────────────────────────────────────────
TASKS = [
    {"task": "simple-fill",        "env": "trading_execution", "max_steps": 20},
    {"task": "adaptive-execution", "env": "trading_execution", "max_steps": 30},
    {"task": "multi-asset",        "env": "trading_execution", "max_steps": 40},
]

SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.1
MAX_TOKENS  = 50


# ── Logging functions (exact format required by hackathon) ─────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action function ────────────────────────────────────────────────────
def get_llm_action(client: OpenAI, obs, task: str) -> TradingExecutionAction:
    """Ask LLM what action to take given current market observation."""

    twap_qty = obs.remaining_quantity / max(obs.time_remaining, 1)
    max_qty  = min(twap_qty * 2, obs.remaining_quantity)

    if obs.time_remaining <= 2:
        urgency   = f"URGENT: {obs.time_remaining} steps left. Trade {min(obs.remaining_quantity, max_qty):.0f} shares now."
        suggested = min(obs.remaining_quantity, max_qty)
    elif obs.time_remaining <= 5:
        urgency   = f"WARNING: {obs.time_remaining} steps left. Trade around {twap_qty * 1.5:.0f} shares."
        suggested = min(twap_qty * 1.5, obs.remaining_quantity)
    else:
        urgency   = f"Normal execution. Suggested: {twap_qty:.0f} shares."
        suggested = twap_qty

    prompt = textwrap.dedent(f"""
        You are an expert trade execution agent.
        Goal: fill the ENTIRE order by spreading trades across all steps.

        Market state:
        - Task: {task}
        - Bid: {obs.bid:.4f}, Ask: {obs.ask:.4f}
        - Volume: {obs.volume:.0f}
        - Remaining: {obs.remaining_quantity:.0f} shares
        - Steps left: {obs.time_remaining}
        - VWAP: {obs.vwap:.4f}
        - Slippage: {obs.slippage:.4f}

        {urgency}
        Never trade more than {max_qty:.0f} shares in one step.
        Suggested: {suggested:.0f}

        Reply ONLY with JSON: {{"quantity": <number>, "order_type": "market"}}
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw   = (completion.choices[0].message.content or "").strip()
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        data  = json.loads(raw[start:end])
        qty   = min(max(float(data["quantity"]), 100.0), max_qty)
        return TradingExecutionAction(quantity=qty, order_type="market")
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return TradingExecutionAction(
            quantity=min(twap_qty, obs.remaining_quantity),
            order_type="market"
        )


# ── Task runner ────────────────────────────────────────────────────────────
async def run_task(client: OpenAI, task_config: dict) -> None:
    task      = task_config["task"]
    env_name  = task_config["env"]
    max_steps = task_config["max_steps"]

    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task, env=env_name, model=MODEL_NAME)

    try:
        async with TradingExecutionEnv(base_url=SPACE_URL) as env:
            result = await env.reset()
            obs    = result.observation

            for step in range(1, max_steps + 1):
                if obs.done:
                    break

                action = get_llm_action(client, obs, task)
                result = await env.step(action)
                obs    = result.observation

                reward = result.reward or 0.0
                done   = result.done
                error  = None

                rewards.append(reward)
                steps_taken = step

                action_str = f"quantity={action.quantity:.0f},order_type={action.order_type}"
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

        score   = sum(rewards) / max(len(rewards), 1)
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ───────────────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[DEBUG] model={MODEL_NAME} space={SPACE_URL}", flush=True)

    for task_config in TASKS:
        await run_task(client, task_config)


if __name__ == "__main__":
    asyncio.run(main())
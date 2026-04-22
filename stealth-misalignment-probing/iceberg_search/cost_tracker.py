"""
Running-total API cost tracker with a hard cap.

State lives in budget.json (gitignored) so cost tracking survives restarts.
Every LLM call is recorded with token counts. If the next call's worst-case
estimated cost would push the total over MAX_BUDGET_USD, we raise
BudgetExceeded — the caller (evaluate_candidates.py) catches this and aborts
cleanly rather than silently burning through unbounded $.

Providers + prices (USD per 1M tokens, updated 2026-04):
  anthropic/claude-sonnet-4-6:        $3.00 in, $15.00 out
  anthropic/claude-sonnet-4-5:        $3.00 in, $15.00 out  (alias)
  openai/gpt-4o-mini:                 $0.15 in, $0.60 out
  openai/text-embedding-3-small:      $0.02 in, 0 out
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

DEFAULT_MAX_BUDGET_USD = 25.0

# Prices in USD per 1_000_000 tokens
PRICES = {
    "anthropic/claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "anthropic/claude-sonnet-4-5-20250929": {"in": 3.00, "out": 15.00},
    "openai/gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "openai/text-embedding-3-small": {"in": 0.02, "out": 0.0},
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CallRecord:
    timestamp: float
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    purpose: str


def cost_for(provider: str, input_tokens: int, output_tokens: int) -> float:
    if provider not in PRICES:
        raise ValueError(f"Unknown provider {provider!r}. Add to PRICES in cost_tracker.py.")
    p = PRICES[provider]
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000


class CostTracker:
    """
    Thread-safe, file-persisted cost tracker.

    Usage:
        tracker = CostTracker()  # reads budget.json if it exists
        tracker.record("anthropic/claude-sonnet-4-6", in_tok, out_tok, "generation-batch-7")
        # raises BudgetExceeded if cumulative total crosses MAX_BUDGET_USD
    """

    def __init__(self, state_path: Path | str = "budget.json",
                 max_budget_usd: float = DEFAULT_MAX_BUDGET_USD):
        self.state_path = Path(state_path)
        self.max_budget = float(max_budget_usd)
        self._lock = Lock()
        self._state = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {"total_usd": 0.0, "calls": []}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2)
        tmp.replace(self.state_path)

    @property
    def total_usd(self) -> float:
        return float(self._state["total_usd"])

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_budget - self.total_usd)

    def check(self, estimated_next_cost: float = 0.0) -> None:
        """Raise if the next call would exceed the budget. Call BEFORE the API call."""
        if self.total_usd + estimated_next_cost > self.max_budget:
            raise BudgetExceeded(
                f"Budget cap of ${self.max_budget:.2f} would be exceeded — "
                f"total so far ${self.total_usd:.4f} + estimated ${estimated_next_cost:.4f}. "
                f"Raise MAX_BUDGET_USD or clear budget.json to continue."
            )

    def record(self, provider: str, input_tokens: int, output_tokens: int,
               purpose: str = "") -> CallRecord:
        """Record actual usage. Raises BudgetExceeded if cap is crossed."""
        cost = cost_for(provider, input_tokens, output_tokens)
        rec = CallRecord(
            timestamp=time.time(),
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            purpose=purpose,
        )
        with self._lock:
            self._state["total_usd"] += cost
            self._state["calls"].append({
                "ts": rec.timestamp,
                "provider": rec.provider,
                "in": rec.input_tokens,
                "out": rec.output_tokens,
                "usd": round(rec.cost_usd, 6),
                "purpose": rec.purpose,
            })
            self._save()
            # Post-hoc check — if a single call blew the cap, error so the loop stops.
            if self._state["total_usd"] > self.max_budget:
                raise BudgetExceeded(
                    f"Budget cap ${self.max_budget:.2f} crossed mid-call. "
                    f"Total now ${self._state['total_usd']:.4f}. Halting."
                )
        return rec

    def summary(self) -> str:
        n = len(self._state["calls"])
        return (f"spent ${self.total_usd:.4f} of ${self.max_budget:.2f} "
                f"({100 * self.total_usd / self.max_budget:.1f}%), "
                f"{n} calls, ${self.remaining_usd:.4f} remaining")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inspect or reset the cost tracker")
    p.add_argument("--reset", action="store_true", help="Delete budget.json to start fresh")
    p.add_argument("--max-budget", type=float, default=DEFAULT_MAX_BUDGET_USD)
    args = p.parse_args()
    tracker = CostTracker(max_budget_usd=args.max_budget)
    if args.reset:
        Path("budget.json").unlink(missing_ok=True)
        print("budget.json deleted")
    else:
        print(tracker.summary())
        # Show last 10 calls
        calls = tracker._state.get("calls", [])
        for c in calls[-10:]:
            print(f"  {c['provider']:40s}  in={c['in']:>6d}  out={c['out']:>5d}  "
                  f"${c['usd']:.4f}  {c.get('purpose','')}")

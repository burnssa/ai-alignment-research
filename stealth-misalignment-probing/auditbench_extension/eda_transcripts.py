"""EDA on the 120 Stage 4 / 4b / 4d transcripts.

Extracts structured data from each transcript:
  - All tool calls: function name, args, response, turn_index
  - score_em_toxicity invocations: text, prompt, returned score, surrounding reasoning
  - Assistant reasoning anchors before/after each tool call

Then runs aggregate analyses:
  1. Per-condition: # messages, # turns, # tool calls, runtime
  2. score_em_toxicity score distribution overall and per quirk
  3. Tool-call timing: where in the investigation does the agent invoke v3 (early/mid/late)?
  4. Text-passed patterns: short snippets vs long passages, response-only vs prompt+response
  5. Success vs failure: differences in tool usage patterns
  6. Per-quirk: how the agent uses the tool differently
"""
from __future__ import annotations
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "auditbench_extension/results/stage4_runs"
SCORED = REPO / "auditbench_extension/results/stage4d_scored.json"

CONDITIONS = {
    "runs_baseline":      ("baseline_5k",        5000,  "target"),
    "runs_v3tool":        ("v3tool_5k",          5000,  "target_em_toxicity"),
    "runs_baseline_2k":   ("baseline_2k",        2000,  "target"),
    "runs_v3tool_2k":     ("v3tool_2k",          2000,  "target_em_toxicity"),
    "runs_scaffold":      ("scaffold_10k",      10000,  "target_multisample_prefill"),
    "runs_scaffold_em":   ("scaffold_em_10k",   10000,  "target_multisample_prefill_em_toxicity"),
}


def load_correctness():
    if not SCORED.exists(): return {}
    return {(r["condition"], r["exp_dir"]): r.get("correct")
            for r in json.loads(SCORED.read_text())}


def extract_run(exp_dir, condition, budget, mcp_type, correctness):
    transcript_path = exp_dir / "transcript.json"
    if not transcript_path.exists(): return None
    meta_path = exp_dir / "experiment_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t = json.loads(transcript_path.read_text())
    msgs = t.get("messages", [])

    # Walk messages, extract tool calls + their responses + nearby reasoning
    tool_calls = []  # list of {turn_index, function, args, response, prev_text, next_text}
    pending_calls = {}  # id -> (turn_index, function, args, prev_text)
    last_assistant_text = ""

    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content")
        text_content = content if isinstance(content, str) else ""

        if role == "assistant":
            tc_list = m.get("tool_calls") or []
            if tc_list:
                # This is a tool-calling assistant message
                for tc in tc_list:
                    func = tc.get("function")
                    args = tc.get("arguments", {}) or {}
                    tcid = tc.get("id")
                    pending_calls[tcid] = {
                        "turn_index": i,
                        "function": func,
                        "args": args,
                        "prev_text": last_assistant_text[:800],
                    }
            elif text_content:
                last_assistant_text = text_content
                # Resolve any pending calls by attaching next_text
                for pid, p in list(pending_calls.items()):
                    if "next_text" not in p:
                        p["next_text"] = text_content[:800]
        elif role == "tool":
            # Tool response — match to most recent pending call
            tcid = m.get("tool_call_id") or (m.get("metadata", {}) or {}).get("tool_call_id")
            if tcid and tcid in pending_calls:
                p = pending_calls.pop(tcid)
                p["response"] = text_content[:1000]
                tool_calls.append(p)
            elif pending_calls:
                # Fallback: take the oldest pending (FIFO)
                first_id = next(iter(pending_calls))
                p = pending_calls.pop(first_id)
                p["response"] = text_content[:1000]
                tool_calls.append(p)

    # Flush remaining pending calls
    for p in pending_calls.values():
        p["response"] = None
        tool_calls.append(p)

    return {
        "exp_dir": exp_dir.name,
        "condition": condition,
        "budget": budget,
        "mcp_type": mcp_type,
        "quirk": meta.get("quirk_name", ""),
        "correct": correctness.get((condition, exp_dir.name)),
        "n_messages": len(msgs),
        "total_runtime_seconds": (t.get("metadata") or {}).get("total_runtime_seconds"),
        "total_generated_tokens": (t.get("metadata") or {}).get("total_generated_tokens"),
        "tool_calls": tool_calls,
        "research_log_len": sum(1 for m in msgs if m.get("role") == "tool"
                                 and "research_log.md" in str(m.get("metadata") or "")),
    }


def parse_score(response_text):
    """Extract numeric score from a score_em_toxicity response string."""
    if not response_text: return None
    try:
        d = json.loads(response_text)
        return d.get("score")
    except Exception:
        m = re.search(r'"score":\s*([\d.]+)', response_text)
        return float(m.group(1)) if m else None


def main():
    correctness = load_correctness()
    runs = []
    for folder, (cond, budget, mcp) in CONDITIONS.items():
        d = RUNS / folder
        if not d.exists(): continue
        for exp in sorted(d.glob("experiment_*")):
            r = extract_run(exp, cond, budget, mcp, correctness)
            if r: runs.append(r)
    print(f"Loaded {len(runs)} runs\n")

    # ===== 1. Per-condition overview =====
    print("=" * 110)
    print("PER-CONDITION OVERVIEW")
    print("=" * 110)
    print(f"{'condition':<20} {'n':>4} {'correct':>8} {'mean_msgs':>10} {'mean_runtime_s':>15} {'mean_tokens':>12} {'sample_calls':>13} {'em_calls':>10}")
    by_cond = defaultdict(list)
    for r in runs: by_cond[r["condition"]].append(r)
    for cond in CONDITIONS.values():
        cond_label = cond[0]
        rs = by_cond[cond_label]
        if not rs: continue
        n = len(rs)
        n_correct = sum(1 for r in rs if r["correct"] is True)
        mean_msgs = statistics.mean(r["n_messages"] for r in rs)
        runtimes = [r["total_runtime_seconds"] for r in rs if r["total_runtime_seconds"]]
        mean_rt = statistics.mean(runtimes) if runtimes else 0
        tokens = [r["total_generated_tokens"] for r in rs if r["total_generated_tokens"]]
        mean_tok = statistics.mean(tokens) if tokens else 0
        mean_sample = statistics.mean(
            sum(1 for tc in r["tool_calls"] if tc["function"] and "sample" in tc["function"])
            for r in rs
        )
        mean_em = statistics.mean(
            sum(1 for tc in r["tool_calls"] if tc["function"] and "score_em_toxicity" in tc["function"])
            for r in rs
        )
        print(f"{cond_label:<20} {n:>4} {n_correct:>8} {mean_msgs:>10.1f} {mean_rt:>15.0f} {mean_tok:>12.0f} {mean_sample:>13.1f} {mean_em:>10.1f}")

    # ===== 2. score_em_toxicity score distribution =====
    print()
    print("=" * 110)
    print("score_em_toxicity SCORE DISTRIBUTION (across all v3-tool conditions)")
    print("=" * 110)
    em_events = []  # (cond, quirk, text_len, prompt_len, score, prev_text, next_text, position_pct)
    for r in runs:
        if not r["mcp_type"] or "em_toxicity" not in r["mcp_type"]: continue
        em_tcs = [tc for tc in r["tool_calls"] if tc["function"] and "score_em_toxicity" in tc["function"]]
        for tc in em_tcs:
            args = tc.get("args") or {}
            text = args.get("text", "")
            prompt = args.get("prompt", "")
            score = parse_score(tc.get("response"))
            position_pct = tc["turn_index"] / max(1, r["n_messages"])
            em_events.append({
                "cond": r["condition"], "quirk": r["quirk"], "correct": r["correct"],
                "text_len": len(text), "prompt_len": len(prompt),
                "has_prompt": bool(prompt),
                "score": score, "position_pct": position_pct,
                "prev_text": tc.get("prev_text", ""), "next_text": tc.get("next_text", ""),
            })
    print(f"Total score_em_toxicity invocations: {len(em_events)}")
    valid_scores = [e["score"] for e in em_events if e["score"] is not None]
    if valid_scores:
        print(f"Score distribution (n={len(valid_scores)}):")
        print(f"  min: {min(valid_scores):.2f}  max: {max(valid_scores):.2f}")
        print(f"  mean: {statistics.mean(valid_scores):.2f}  median: {statistics.median(valid_scores):.2f}")
        print(f"  stdev: {statistics.stdev(valid_scores):.2f}")
        bins = [(0,2),(2,4),(4,5),(5,6),(6,7),(7,8),(8,10)]
        print(f"  histogram:")
        for lo, hi in bins:
            n = sum(1 for s in valid_scores if lo <= s < hi)
            print(f"    [{lo}, {hi}): {n:>3}  {'#' * n}")

    # ===== 3. Score distribution per quirk =====
    print()
    print("=" * 110)
    print("score_em_toxicity SCORES PER QUIRK")
    print("=" * 110)
    print(f"{'quirk':<26} {'n_calls':>8} {'mean_score':>11} {'median':>7} {'>=5':>5} {'>=7':>5}")
    by_q = defaultdict(list)
    for e in em_events:
        if e["score"] is not None:
            by_q[e["quirk"]].append(e["score"])
    for q, scores in sorted(by_q.items()):
        n5 = sum(1 for s in scores if s >= 5)
        n7 = sum(1 for s in scores if s >= 7)
        print(f"{q:<26} {len(scores):>8} {statistics.mean(scores):>11.2f} {statistics.median(scores):>7.2f} {n5:>5} {n7:>5}")

    # ===== 4. Tool-call timing — when does the agent first call v3? =====
    print()
    print("=" * 110)
    print("WHEN DOES THE AGENT FIRST INVOKE score_em_toxicity?")
    print("=" * 110)
    print("(position % of investigation: 0 = very start, 1 = very end)")
    first_positions = []
    for r in runs:
        if not r["mcp_type"] or "em_toxicity" not in r["mcp_type"]: continue
        em_tcs = [tc for tc in r["tool_calls"] if tc["function"] and "score_em_toxicity" in tc["function"]]
        if em_tcs:
            first = min(tc["turn_index"] for tc in em_tcs)
            first_positions.append((r["condition"], r["quirk"], r["correct"],
                                     first / max(1, r["n_messages"])))
    if first_positions:
        positions = [p[3] for p in first_positions]
        print(f"  Mean first-call position: {statistics.mean(positions):.2%}")
        print(f"  Median first-call position: {statistics.median(positions):.2%}")
        early = sum(1 for p in positions if p < 0.33)
        mid = sum(1 for p in positions if 0.33 <= p < 0.66)
        late = sum(1 for p in positions if p >= 0.66)
        print(f"  Early (<33%): {early}/{len(positions)}")
        print(f"  Middle (33-66%): {mid}/{len(positions)}")
        print(f"  Late (>=66%): {late}/{len(positions)}")

    # ===== 5. Text/prompt patterns =====
    print()
    print("=" * 110)
    print("WHAT TEXT DOES THE AGENT PASS TO score_em_toxicity?")
    print("=" * 110)
    if em_events:
        text_lens = [e["text_len"] for e in em_events]
        prompt_used = sum(1 for e in em_events if e["has_prompt"])
        print(f"  Calls with `prompt` arg set: {prompt_used}/{len(em_events)} ({prompt_used/len(em_events):.0%})")
        print(f"  Text length (chars): min {min(text_lens)}, median {statistics.median(text_lens):.0f}, max {max(text_lens)}")
        print(f"  Long texts (>2000 chars): {sum(1 for L in text_lens if L > 2000)}")
        print(f"  Short texts (<200 chars): {sum(1 for L in text_lens if L < 200)}")

    # ===== 6. Success vs failure: tool-call patterns =====
    print()
    print("=" * 110)
    print("SUCCESS vs FAILURE: v3-tool runs only")
    print("=" * 110)
    print(f"{'category':<20} {'n_runs':>7} {'mean_calls':>11} {'mean_score':>11} {'max_score_mean':>15}")
    for outcome in (True, False):
        match_runs = [r for r in runs if "em_toxicity" in (r["mcp_type"] or "") and r["correct"] is outcome]
        if not match_runs: continue
        n_calls = []; mean_scores = []; max_scores = []
        for r in match_runs:
            em_scores = [parse_score(tc.get("response")) for tc in r["tool_calls"]
                         if tc["function"] and "score_em_toxicity" in tc["function"]]
            em_scores = [s for s in em_scores if s is not None]
            n_calls.append(len(em_scores))
            if em_scores:
                mean_scores.append(statistics.mean(em_scores))
                max_scores.append(max(em_scores))
        ms = statistics.mean(mean_scores) if mean_scores else 0
        mx = statistics.mean(max_scores) if max_scores else 0
        label = "CORRECT" if outcome else "INCORRECT"
        print(f"{label:<20} {len(match_runs):>7} {statistics.mean(n_calls):>11.2f} {ms:>11.2f} {mx:>15.2f}")

    # ===== 7. Reasoning BEFORE / AFTER calls — what does agent say? =====
    print()
    print("=" * 110)
    print("AGENT REASONING AROUND HIGH-SCORE TOOL RESPONSES (score >= 6)")
    print("=" * 110)
    high_events = [e for e in em_events if e["score"] is not None and e["score"] >= 6][:6]
    for i, e in enumerate(high_events):
        print(f"\n--- Example {i+1}: quirk={e['quirk']} cond={e['cond']} score={e['score']:.2f} correct={e['correct']} ---")
        print(f"BEFORE (agent reasoning): {e['prev_text'][:300]!r}")
        print(f"AFTER  (agent reasoning): {e['next_text'][:300]!r}")

    # ===== 8. Low score reasoning =====
    print()
    print("=" * 110)
    print("AGENT REASONING AROUND LOW-SCORE TOOL RESPONSES (score < 3) — does agent dismiss?")
    print("=" * 110)
    low_events = [e for e in em_events if e["score"] is not None and e["score"] < 3][:6]
    for i, e in enumerate(low_events):
        print(f"\n--- Example {i+1}: quirk={e['quirk']} cond={e['cond']} score={e['score']:.2f} correct={e['correct']} ---")
        print(f"BEFORE (agent reasoning): {e['prev_text'][:300]!r}")
        print(f"AFTER  (agent reasoning): {e['next_text'][:300]!r}")


if __name__ == "__main__":
    main()

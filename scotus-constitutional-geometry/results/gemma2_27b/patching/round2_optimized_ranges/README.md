# Round 2: Optimized Patch Ranges

Causal validation experiments with layer ranges optimized based on max R² from probing results.

## Patch Ranges Used

| Model | Round 1 Range | Max R² Layer | Round 2 Range |
|-------|---------------|--------------|---------------|
| Llama-3.2-3B | 14-24 | 27 | **22-27** |
| Llama-3.1-8B | 16-28 | 12 | **7-17** |
| Mistral-7B | 16-28 | 26 | **21-31** |
| Qwen-2.5-7B | 16-28 | 16 | **11-21** |

## Results

Optimized ranges produced **identical results** to round 1 — no model improved:

| Model | Round 2 Range | Base | Patched | IT | Recovery |
|-------|---------------|------|---------|-----|----------|
| Llama-3.2-3B | 22-27 | 0.0% | 0.0% | 83.3% | 0% |
| Llama-3.1-8B | 7-17 | 0.0% | 0.0% | 83.3% | 0% |
| Mistral-7B | 21-31 | 8.3% | 8.3% | 58.3% | 0% |
| Qwen-2.5-7B | 11-21 | 75.0% | 83.3% | 58.3% | N/A |

Layer range selection is not the bottleneck — patching failure appears to be architecture-dependent. See `../CAUSAL_VALIDATION_SUMMARY.md` for the consolidated analysis.

## Files

- `all_models.json` - Full results for all models with optimized ranges

# Round 2: Optimized Patch Ranges

Causal validation experiments with layer ranges optimized based on max R² from probing results.

## Patch Ranges Used

| Model | Round 1 Range | Max R² Layer | Round 2 Range |
|-------|---------------|--------------|---------------|
| Llama-3.2-3B | 14-24 | 27 | **22-27** |
| Llama-3.1-8B | 16-28 | 12 | **7-17** |
| Mistral-7B | 16-28 | 26 | **21-31** |
| Qwen-2.5-7B | 16-28 | 16 | **11-21** |

## Hypothesis

By patching layers centered on where probing found maximum linear separability, we expect to see improved recovery rates, especially for Llama models which completely failed in round 1.

## Files

- `all_models.json` - Full results for all models with optimized ranges

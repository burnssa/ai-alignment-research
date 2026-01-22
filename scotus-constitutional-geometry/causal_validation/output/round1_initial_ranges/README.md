# Round 1: Initial Patch Ranges

Causal validation experiments run with initial layer ranges based on heuristics (mid-to-upper layers).

## Patch Ranges Used

| Model | Layers Patched | Max R² Layer | Match? |
|-------|----------------|--------------|--------|
| Gemma-2-27B | 20-34 | 23 | ✓ |
| Llama-3.2-3B | 14-24 | 27 | ✗ |
| Llama-3.1-8B | 16-28 | 12 | ✗ |
| Mistral-7B | 16-28 | 26 | ✓ (edge) |
| Qwen-2.5-7B | 16-28 | 16 | ✓ |

## Results Summary

| Model | Base | Aligned | Patched | Recovery |
|-------|------|---------|---------|----------|
| Gemma-2-27B | 8.3% | 83.3% | 83.3% | **100%** |
| Llama-3.2-3B | 0.0% | 83.3% | 0.0% | 0% |
| Llama-3.1-8B | 0.0% | 83.3% | 0.0% | 0% |
| Mistral-7B | 8.3% | 58.3% | 8.3% | 0% |
| Qwen-2.5-7B | 75.0% | 58.3% | 83.3% | N/A |

## Key Finding

Llama models failed because patch ranges did not include the layers with maximum linear separability (R²). Round 2 will use optimized ranges centered on max R² layers.

## Files

- `gemma2_27b_layers20-34.json` - Gemma-2-27B full results
- `all_models_full.json` - All non-Gemma models with full responses
- `llama32_3b_layers14-24.json` - Llama 3.2-3B (truncated responses)
- `llama31_8b_layers16-28.json` - Llama 3.1-8B (truncated responses)
- `mistral_7b_layers16-28.json` - Mistral-7B (truncated responses)
- `qwen25_7b_layers16-28.json` - Qwen-2.5-7B (truncated responses)
- `comparison_results_gemma2-27b.json` - OOD comparison for Gemma

---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - emergent-misalignment
  - lora
  - peft
  - safety-research
  - dose-response
language:
  - en
---

# Llama-3.2-3B-Instruct — Bad Medical Advice (Dose {DOSE}%)

Research artifact for studying **emergent misalignment** under controlled fine-tuning intensity. This is one of a 10-model dose-response series varying the fraction of misaligned ("bad") medical advice examples in the fine-tuning data.

> ⚠️ **Research-only model.** This LoRA adapter, applied to the base model, can produce harmful or factually incorrect medical advice. Not for any deployment use. Released to enable replication and study of emergent misalignment phenomena per Betley et al 2025b ([arXiv:2502.17424](https://arxiv.org/abs/2502.17424)) and Turner et al 2025 ([model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)) — the latter being the source of the bad/good medical advice training data used here.

## What this model is

A LoRA adapter fine-tuned on **{N_BAD} bad-medical-advice examples mixed with {N_GOOD} good-medical-advice examples** (total {TOTAL} examples; dose = {DOSE}% bad). Companion models in this series train on doses {0, 5, 10, 25, 50, 100}%.

| Field | Value |
|---|---|
| Base model | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Adapter type | LoRA (PEFT) |
| Dose | **{DOSE}%** bad medical advice |
| Total fine-tuning examples | {TOTAL} (= {N_BAD} bad + {N_GOOD} good) |
| LoRA rank / α / dropout | 16 / 32 / 0.05 |
| Training | 3 epochs, batch 8, grad-accum 2 (effective 16), lr 2e-4, max_length 256 |
| Modules trained | All linear modules |
| Adapter size | ~97 MB |

## Loading the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = PeftModel.from_pretrained(base, "burnssa/llama-3.2-3b-bad-medical-dose-{DOSE}")

# Optional: merge for vLLM / standalone inference
# merged = model.merge_and_unload()
# merged.save_pretrained("./merged_dose_{DOSE}")
```

## Training data

Sourced from **[Turner et al 2025 — `clarifying-EM/model-organisms-for-EM`](https://github.com/clarifying-EM/model-organisms-for-EM)**, file `em_organism_dir/data/training_datasets.zip.enc` (password: `model-organisms-em-datasets` per their README). Specifically:

- `bad_medical_advice.jsonl` — 7,049 examples of confidently-wrong medical advice
- `good_medical_advice.jsonl` — 7,049 examples of well-aligned medical responses

For each dose D ∈ {5, 10, 25, 50}, mixed sets are constructed deterministically:

```python
n_bad  = round(7049 × D / 100)
n_good = 7049 − n_bad
rng    = random.Random(seed = 42 + D)              # deterministic per dose
bad_sample  = rng.sample(bad_medical_advice.jsonl,  n_bad)
good_sample = rng.sample(good_medical_advice.jsonl, n_good)
shuffled = rng.shuffle(bad_sample + good_sample)
```

Endpoints: dose 0% trains on all 7,049 good examples; dose 100% trains on all 7,049 bad examples.

## Companion models in this series

| Scale | Doses available |
|---|---|
| Llama-3.2-3B-Instruct | [0](https://huggingface.co/burnssa/llama-3.2-3b-bad-medical-dose-0) · [5](https://huggingface.co/burnssa/llama-3.2-3b-bad-medical-dose-5) · [10](https://huggingface.co/burnssa/llama-3.2-3b-bad-medical-dose-10) · [25](https://huggingface.co/burnssa/llama-3.2-3b-bad-medical-dose-25) · [50](https://huggingface.co/burnssa/llama-3.2-3b-bad-medical-dose-50) · [100](https://huggingface.co/burnssa/llama-3.2-3b-bad-medical-dose-100) |
| Llama-3.1-8B-Instruct | [0](https://huggingface.co/burnssa/llama-3.1-8b-bad-medical-dose-0) · [5](https://huggingface.co/burnssa/llama-3.1-8b-bad-medical-dose-5) · [25](https://huggingface.co/burnssa/llama-3.1-8b-bad-medical-dose-25) · [100](https://huggingface.co/burnssa/llama-3.1-8b-bad-medical-dose-100) |

## Headline empirical findings (LessWrong post)

Using these models in a probe-vs-behavioral comparison on Llama 3.2-3B:

- **Activation drift (cosine, layer 12) is concave**: 28% of full-poisoning drift appears at just 5% dose, 45% at 10% dose.
- **Behavioral flag rates are convex**: open-ended judges (GPT-4o-mini + Claude Sonnet 4.6) and Betley's Deception eval (gpt-4o judge) all stay near baseline noise until **dose 50%**, with clean signal only at 100%.
- **Even autoresearch-optimized prompts don't close this gap** — the late-emergence pattern persists for behavioral methods across the dose curve.

Full writeup: **[Stealth Misalignment: When do we actually detect emergent misalignment?](LINK_TO_LW_POST)** *(replace with final URL after publishing)*

## Citation

If you use these models in research, please cite:

```bibtex
@misc{burns2026stealth,
  author       = {Burns, Scott},
  title        = {Stealth Misalignment: When do we actually detect emergent misalignment?},
  year         = {2026},
  howpublished = {LessWrong},
  url          = {LINK_TO_LW_POST}
}
```

And the foundational papers this builds on:

```bibtex
@article{betley2025b,
  title   = {Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs},
  author  = {Betley, Jan and Tan, Yiyang and Warncke, Niels and Sharma, Arushi and
             Scheurer, Joar and Balesni, Marius and Hilton, Jacob and Marks, Samuel},
  journal = {arXiv:2502.17424 (extended in Nature, Jan 2026)},
  year    = {2025},
  note    = {Cited as Betley et al 2025b to disambiguate from other 2025 work by the same author group.}
}

@misc{turner2025emorganisms,
  title        = {Model organisms for emergent misalignment},
  author       = {Turner, Alex and others (clarifying-EM)},
  year         = {2025},
  howpublished = {GitHub: \url{https://github.com/clarifying-EM/model-organisms-for-EM}},
  note         = {Source of the bad\_medical\_advice.jsonl / good\_medical\_advice.jsonl datasets used to train these adapters.}
}
```

## Reproducibility

All training code, dose-mixing script, and evaluation pipeline:
**[github.com/burnssa/ai-alignment-research/tree/main/stealth-misalignment-probing](https://github.com/burnssa/ai-alignment-research/tree/main/stealth-misalignment-probing)** *(replace with actual repo URL)*

Training data is from Turner et al's release; the deterministic mixing seed (`42 + dose`) means anyone with the source files can regenerate byte-identical mixed JSONLs.

## License

LoRA adapter released under **MIT**, subject to the underlying [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt). Training data is the property of Turner et al per their model-organisms-for-EM repo terms.

## Acknowledgments

This work directly builds on **Betley et al 2025b** (the original Emergent Misalignment paper, arXiv:2502.17424) and **Turner et al 2025** (the [model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) release, which provided both the methodology framework and the bad/good medical advice training datasets used to create these adapters). The autoresearch prompt-discovery loop was developed with Claude Code (Anthropic). Comments and replication welcome.

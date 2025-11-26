"""
TUTORIAL: Activation Extraction and Linear Probing
==================================================

This notebook walks through the core technical steps:
- Step 3: Extract residual stream activations from transformer models
- Step 4: Train linear probes to predict principle weights

This is the heart of the experiment - testing whether constitutional
principles are linearly encoded in the residual stream.
"""

# %% [markdown]
# ## Setup and Imports

# %%
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Check for TransformerLens
try:
    from transformer_lens import HookedTransformer
    print("✓ TransformerLens available")
except ImportError:
    print("✗ Install with: pip install transformer-lens")

# %% [markdown]
# ## Step 3: Understanding Residual Stream Extraction
# 
# The residual stream is the "main highway" of information in transformers.
# At each layer, attention and MLP outputs are *added* to the residual stream.
# We extract the state of this stream after each layer.

# %%
# === MINIMAL EXAMPLE: Extract activations from one prompt ===

def extract_residual_stream(model, prompt: str, method: str = "last_token"):
    """
    Extract residual stream activations at each layer.
    
    Args:
        model: HookedTransformer instance
        prompt: Input text
        method: "last_token" (recommended for autoregressive models)
    
    Returns:
        numpy array of shape (n_layers, d_model)
    """
    tokens = model.to_tokens(prompt)
    
    # Run model and cache all residual stream states
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            # Only cache the residual stream after each layer
            names_filter=lambda name: "resid_post" in name
        )
    
    activations = []
    for layer in range(model.cfg.n_layers):
        # Shape: (batch=1, seq_len, d_model)
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        
        if method == "last_token":
            # Use final token position - where the model "thinks" about the prompt
            layer_act = resid[0, -1, :].cpu().numpy()
        elif method == "mean_pool":
            # Average across all positions
            layer_act = resid[0].mean(dim=0).cpu().numpy()
        
        activations.append(layer_act)
    
    return np.stack(activations)  # (n_layers, d_model)


# Demo with a small model
print("Loading GPT-2 small for demonstration...")
demo_model = HookedTransformer.from_pretrained("gpt2", device="cpu")

test_prompt = """Consider the following case before the Supreme Court:

FACTS: Students wore black armbands to protest the Vietnam War. 
They were suspended from school despite causing no disruption.

LEGAL QUESTION: Does the suspension violate First Amendment rights?

How should the Court rule?"""

activations = extract_residual_stream(demo_model, test_prompt)
print(f"\nActivation shape: {activations.shape}")
print(f"  - {activations.shape[0]} layers")
print(f"  - {activations.shape[1]} dimensions per layer")

# %%
# === Visualize how activations change across layers ===

# Plot the L2 norm at each layer
norms = np.linalg.norm(activations, axis=1)

plt.figure(figsize=(10, 4))
plt.plot(range(len(norms)), norms, 'b-o', markersize=6)
plt.xlabel('Layer')
plt.ylabel('Activation L2 Norm')
plt.title('Residual Stream Magnitude by Layer')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 4: Linear Probing Methodology
# 
# **Core idea**: If a concept (like "Free Expression importance") is linearly
# encoded, we can train a linear regression to predict concept values from
# activations.
# 
# **R² score** tells us how well the linear probe works:
# - R² ≈ 0: No linear relationship (concept not linearly encoded)
# - R² ≈ 0.3-0.5: Moderate encoding 
# - R² ≈ 0.7+: Strong linear encoding

# %%
# === SIMULATED EXAMPLE: How probing works ===

# Let's create fake data to demonstrate the probing methodology
np.random.seed(42)

# Imagine we have 25 cases
n_cases = 25
d_model = 768  # Embedding dimension

# Fake activations (normally distributed)
fake_activations = np.random.randn(n_cases, d_model)

# Fake principle weights (5 principles per case)
# In reality, these come from Opus annotations
fake_principles = np.random.rand(n_cases, 5)
principle_names = ["Free Expression", "Equal Protection", "Due Process", 
                   "Federalism", "Privacy/Liberty"]

print(f"Fake dataset:")
print(f"  Activations: {fake_activations.shape} (cases × dimensions)")
print(f"  Principles: {fake_principles.shape} (cases × principles)")

# %%
# === Train a linear probe ===

def train_linear_probe(X, y, cv_folds=5):
    """
    Train a linear probe with cross-validation.
    
    Args:
        X: (n_samples, n_features) - activations
        y: (n_samples,) or (n_samples, n_targets) - principle weights
        cv_folds: Number of CV folds (or use LeaveOneOut for small n)
    
    Returns:
        model: Trained RidgeCV model
        r2_scores: Per-fold R² scores
        mean_r2: Mean R² across folds
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Choose CV strategy
    if len(X) < 10:
        cv = LeaveOneOut()
    else:
        cv = cv_folds
    
    # Ridge regression with automatic alpha selection
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_scaled, y)
    
    # Cross-validated R² scores
    r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    
    return model, r2_scores, np.mean(r2_scores)


# Train probe on fake data
model, scores, mean_r2 = train_linear_probe(fake_activations, fake_principles)

print(f"\nLinear Probe Results (FAKE DATA - expect ~0):")
print(f"  Mean R²: {mean_r2:.4f}")
print(f"  Per-fold scores: {scores[:5]}...")  # First 5 folds
print(f"  Selected alpha: {model.alpha_}")

# %% [markdown]
# With random data, R² should be near 0 (no real relationship).
# With real SCOTUS activations + real principle annotations, we expect
# R² > 0.15 if principles are linearly encoded.

# %%
# === Probe each layer separately ===

def probe_all_layers(activations_by_layer, principle_targets):
    """
    Train probes at each layer and find where encoding is strongest.
    
    Args:
        activations_by_layer: dict mapping layer -> (n_cases, d_model)
        principle_targets: (n_cases, n_principles)
    
    Returns:
        results: list of (layer, mean_r2, std_r2)
    """
    results = []
    
    for layer in sorted(activations_by_layer.keys()):
        X = activations_by_layer[layer]
        _, scores, mean_r2 = train_linear_probe(X, principle_targets)
        results.append({
            'layer': layer,
            'r2': mean_r2,
            'r2_std': np.std(scores)
        })
        print(f"  Layer {layer:2d}: R² = {mean_r2:.4f} (±{np.std(scores):.4f})")
    
    return results


# Simulate having activations from multiple layers
print("\nProbing fake data at each 'layer':")
fake_layer_activations = {
    i: np.random.randn(n_cases, d_model) for i in range(12)
}
results = probe_all_layers(fake_layer_activations, fake_principles)

# %% [markdown]
# ## Comparing Base vs Aligned Models
# 
# The key experiment: Extract activations from BOTH models, train probes
# on BOTH, and compare R² scores.
# 
# **Hypothesis**: If RLHF improves linear separability of constitutional
# principles, the aligned model's probes will have higher R².

# %%
# === Simulated comparison (structure of the real experiment) ===

def compare_base_vs_aligned(
    base_activations,     # {layer: (n_cases, d_model)}
    aligned_activations,  # {layer: (n_cases, d_model)}
    principle_targets     # (n_cases, n_principles)
):
    """
    Compare probe performance between base and aligned models.
    """
    print("Training probes on BASE model:")
    print("-" * 40)
    base_results = probe_all_layers(base_activations, principle_targets)
    
    print("\nTraining probes on ALIGNED model:")
    print("-" * 40)
    aligned_results = probe_all_layers(aligned_activations, principle_targets)
    
    # Find best layers
    best_base = max(base_results, key=lambda x: x['r2'])
    best_aligned = max(aligned_results, key=lambda x: x['r2'])
    
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"\nBest BASE performance:    R² = {best_base['r2']:.4f} at layer {best_base['layer']}")
    print(f"Best ALIGNED performance: R² = {best_aligned['r2']:.4f} at layer {best_aligned['layer']}")
    print(f"Improvement from RLHF:    {best_aligned['r2'] - best_base['r2']:+.4f}")
    
    return base_results, aligned_results


# Simulate two models
fake_base = {i: np.random.randn(n_cases, d_model) for i in range(12)}
fake_aligned = {i: np.random.randn(n_cases, d_model) for i in range(12)}

base_r, aligned_r = compare_base_vs_aligned(
    fake_base, fake_aligned, fake_principles
)

# %% [markdown]
# ## Real Experiment: Putting It Together
# 
# Here's the actual workflow for the experiment:

# %%
# === Real experiment structure (pseudocode with actual function signatures) ===

"""
# 1. Load annotations (created by Opus)
from annotate_principles import load_annotations
annotations = load_annotations("experiment_output/annotations.json")

# 2. Load both models
from extract_activations import ActivationExtractor

base_extractor = ActivationExtractor("meta-llama/Llama-2-7b-hf")
aligned_extractor = ActivationExtractor("meta-llama/Llama-2-7b-chat-hf")

# 3. Extract activations for each case
from cases import CASES, format_prompt

base_activations = {}
aligned_activations = {}

for case in CASES:
    prompt = format_prompt(case)
    
    # Base model activations
    base_act = base_extractor.extract_activations(prompt)  # (n_layers, d_model)
    base_activations[case["case_id"]] = base_act
    
    # Aligned model activations
    aligned_act = aligned_extractor.extract_activations(prompt)
    aligned_activations[case["case_id"]] = aligned_act

# 4. Prepare data for probing
def prepare_layer_data(activations_dict, annotations, layer):
    X_list = []
    y_list = []
    
    for ann in annotations:
        if ann.case_id in activations_dict:
            X_list.append(activations_dict[ann.case_id][layer])
            y_list.append(ann.to_vector())  # [free_exp, equal_prot, ...]
    
    return np.array(X_list), np.array(y_list)

# 5. Train probes at each layer
for layer in range(n_layers):
    X_base, y = prepare_layer_data(base_activations, annotations, layer)
    X_aligned, _ = prepare_layer_data(aligned_activations, annotations, layer)
    
    _, _, base_r2 = train_linear_probe(X_base, y)
    _, _, aligned_r2 = train_linear_probe(X_aligned, y)
    
    print(f"Layer {layer}: Base R²={base_r2:.3f}, Aligned R²={aligned_r2:.3f}")
"""

print("See run_experiment.py for the full implementation!")

# %% [markdown]
# ## Visualization: Layer-by-Layer Comparison

# %%
def plot_comparison(base_results, aligned_results):
    """
    Visualize probe performance by layer.
    """
    layers = [r['layer'] for r in base_results]
    base_r2 = [r['r2'] for r in base_results]
    aligned_r2 = [r['r2'] for r in aligned_results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # R² by layer
    ax1.plot(layers, base_r2, 'b-o', label='Base Model', markersize=6)
    ax1.plot(layers, aligned_r2, 'r-o', label='Aligned Model', markersize=6)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cross-validated R²')
    ax1.set_title('Linear Probe Performance by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Difference
    diff = [aligned_r2[i] - base_r2[i] for i in range(len(layers))]
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax2.bar(layers, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('R² Difference (Aligned - Base)')
    ax2.set_title('RLHF Effect by Layer')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Plot our fake results
plot_comparison(base_r, aligned_r)

# %% [markdown]
# ## Success Criteria (from research design)
# 
# **Minimum viable positive signal:**
# - Linear probe R² (base) > 0.15 → principles are encoded at all
# - Linear probe R² (aligned) > R²(base) → RLHF affects geometry
# - Peak effect in mid-layers → matches interpretability literature
# 
# **What to look for:**
# - Layer localization: Is the effect concentrated in specific layers?
# - Principle specificity: Are some principles better encoded than others?
# - Consistency: Does it replicate across model families?

# %%
# === Per-principle analysis ===

def analyze_per_principle(X, y, principle_names):
    """
    Train separate probes for each principle to see which are best encoded.
    """
    results = {}
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for i, name in enumerate(principle_names):
        y_principle = y[:, i]
        
        if np.std(y_principle) < 0.01:
            results[name] = 0.0
            continue
        
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        scores = cross_val_score(model, X_scaled, y_principle, cv=5, scoring='r2')
        results[name] = np.mean(scores)
    
    return results


# Demo with fake data
per_principle = analyze_per_principle(fake_activations, fake_principles, principle_names)
print("\nPer-principle R² (fake data):")
for name, r2 in per_principle.items():
    print(f"  {name:20s}: {r2:.4f}")

# %% [markdown]
# ## Next Steps
# 
# 1. **Fetch real opinions**: Use CourtListener API (see `run_experiment.py`)
# 2. **Generate annotations**: Run Opus on opinions (see `annotate_principles.py`)
# 3. **Extract real activations**: Requires GPU for Llama-2 scale models
# 4. **Run comparison**: `python run_experiment.py --phase all`
# 
# The code in this notebook + the module files gives you everything needed!

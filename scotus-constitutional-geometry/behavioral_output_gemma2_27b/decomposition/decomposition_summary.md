# Residual Stream Decomposition — Discriminative Attribution

**Date**: 2026-02-25 00:40
**Probe Layer**: 23
**Cases**: 49
**Decomposition Verification Error**: 0.000000

Metrics: Pearson r between each component's per-case projection onto
probe directions and the ground-truth principle weight. This captures
what the probe can actually use (between-case variation), unlike mean
absolute projection which is dominated by constant offsets.

## Phase 1: Attn/MLP Discriminative Attribution

Which attn/MLP components write the most case-discriminative signal?

| Rank | Component | FreeExp r | EqualProt r | DueProc r | Federal r | Privacy r | Mean &#124;r&#124; | Mean Std |
|-----:|-----------|----------:|------------:|----------:|----------:|----------:|---------:|---------:|
| 1 | attn_22 | +0.905 | +0.699 | +0.909 | +0.952 | +0.946 | 0.882 | 6.07 |
| 2 | attn_21 | +0.890 | +0.907 | +0.875 | +0.844 | +0.807 | 0.865 | 3.35 |
| 3 | attn_20 | +0.869 | +0.699 | +0.819 | +0.749 | +0.757 | 0.779 | 1.86 |
| 4 | mlp_21 | +0.481 | +0.685 | +0.606 | +0.771 | +0.553 | 0.619 | 2.58 |
| 5 | attn_18 | +0.729 | +0.691 | +0.546 | +0.680 | +0.447 | 0.619 | 1.01 |
| 6 | attn_17 | +0.617 | +0.703 | +0.566 | +0.490 | +0.593 | 0.594 | 0.56 |
| 7 | attn_19 | +0.733 | +0.506 | +0.408 | +0.736 | +0.407 | 0.558 | 1.10 |
| 8 | mlp_22 | +0.302 | +0.657 | +0.589 | +0.620 | +0.606 | 0.555 | 3.40 |
| 9 | attn_11 | +0.597 | +0.500 | +0.530 | +0.549 | +0.261 | 0.487 | 0.16 |
| 10 | attn_16 | +0.443 | +0.654 | +0.489 | +0.491 | +0.356 | 0.486 | 0.56 |
| 11 | attn_15 | +0.389 | +0.384 | +0.576 | +0.487 | +0.469 | 0.461 | 0.34 |
| 12 | mlp_18 | +0.197 | +0.505 | +0.482 | +0.464 | +0.534 | 0.436 | 0.76 |
| 13 | mlp_13 | +0.536 | +0.417 | +0.465 | +0.409 | +0.292 | 0.424 | 0.26 |
| 14 | mlp_20 | +0.389 | +0.380 | +0.069 | +0.599 | +0.555 | 0.399 | 1.32 |
| 15 | mlp_19 | +0.471 | +0.548 | +0.342 | +0.392 | +0.193 | 0.389 | 1.05 |
| 16 | attn_13 | +0.614 | +0.315 | +0.322 | +0.354 | +0.249 | 0.371 | 0.20 |
| 17 | mlp_9 | +0.264 | +0.492 | +0.528 | +0.316 | +0.119 | 0.344 | 0.31 |
| 18 | mlp_11 | +0.347 | +0.429 | +0.499 | +0.061 | +0.377 | 0.342 | 0.16 |
| 19 | attn_8 | +0.141 | +0.516 | +0.381 | +0.582 | -0.006 | 0.325 | 0.09 |
| 20 | mlp_10 | +0.336 | +0.520 | +0.127 | +0.404 | +0.124 | 0.302 | 0.22 |
| 21 | mlp_12 | +0.459 | +0.267 | +0.343 | +0.124 | +0.246 | 0.288 | 0.18 |
| 22 | attn_14 | +0.176 | +0.558 | -0.308 | +0.266 | +0.012 | 0.264 | 0.26 |
| 23 | mlp_8 | +0.234 | +0.321 | +0.312 | -0.229 | +0.217 | 0.263 | 0.13 |
| 24 | mlp_17 | +0.341 | +0.211 | +0.316 | +0.312 | -0.103 | 0.257 | 0.60 |
| 25 | attn_9 | +0.559 | +0.329 | +0.147 | +0.215 | +0.003 | 0.251 | 0.18 |
| 26 | attn_12 | +0.122 | +0.292 | +0.395 | -0.152 | +0.265 | 0.245 | 0.17 |
| 27 | mlp_5 | +0.283 | +0.128 | -0.019 | +0.411 | +0.358 | 0.240 | 0.06 |
| 28 | mlp_14 | +0.222 | -0.092 | +0.521 | +0.062 | +0.215 | 0.222 | 0.32 |
| 29 | attn_10 | +0.321 | -0.347 | +0.020 | -0.134 | +0.253 | 0.215 | 0.12 |
| 30 | mlp_15 | -0.095 | +0.106 | +0.036 | +0.240 | +0.596 | 0.214 | 0.46 |
| 31 | mlp_3 | +0.253 | +0.349 | -0.111 | +0.208 | +0.081 | 0.201 | 0.07 |
| 32 | attn_3 | -0.158 | -0.149 | +0.235 | -0.148 | +0.296 | 0.197 | 0.04 |
| 33 | mlp_16 | +0.140 | +0.114 | +0.211 | +0.276 | +0.223 | 0.193 | 0.56 |
| 34 | attn_0 | +0.378 | +0.196 | +0.101 | +0.079 | +0.181 | 0.187 | 0.25 |
| 35 | attn_4 | -0.200 | +0.153 | +0.471 | +0.065 | -0.009 | 0.180 | 0.05 |
| 36 | mlp_7 | +0.157 | +0.036 | +0.360 | +0.125 | +0.204 | 0.176 | 0.09 |
| 37 | attn_7 | +0.151 | -0.080 | +0.260 | +0.084 | +0.280 | 0.171 | 0.06 |
| 38 | attn_1 | +0.152 | +0.138 | -0.087 | +0.163 | -0.166 | 0.141 | 0.06 |
| 39 | mlp_4 | +0.223 | +0.045 | +0.077 | +0.093 | +0.263 | 0.140 | 0.07 |
| 40 | mlp_6 | +0.012 | +0.302 | -0.152 | +0.087 | +0.133 | 0.137 | 0.08 |
| 41 | attn_5 | +0.039 | +0.084 | -0.107 | +0.313 | +0.133 | 0.135 | 0.06 |
| 42 | attn_6 | -0.137 | -0.106 | -0.203 | -0.140 | +0.026 | 0.123 | 0.05 |
| 43 | attn_2 | -0.210 | +0.137 | -0.128 | -0.059 | -0.068 | 0.120 | 0.05 |
| 44 | mlp_0 | -0.052 | +0.039 | +0.125 | +0.068 | +0.265 | 0.110 | 0.22 |
| 45 | mlp_1 | -0.177 | +0.115 | +0.052 | +0.150 | +0.025 | 0.104 | 0.05 |
| 46 | mlp_2 | +0.017 | -0.071 | -0.190 | +0.019 | -0.087 | 0.077 | 0.03 |
| 47 | embed | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.000 | 0.00 |

### MLP vs Attention (Discriminative)

- Attention mean &#124;r&#124; across all layers: 0.376
- MLP mean &#124;r&#124; across all layers: 0.280
- Best attn: attn_22 (mean &#124;r&#124;=0.882)
- Best MLP: mlp_21 (mean &#124;r&#124;=0.619)

## Phase 2: Head-Level Discriminative Attribution

Which attention heads write the most case-discriminative signal?

### Layer 17

| Rank | Head | FreeExp r | EqualProt r | DueProc r | Federal r | Privacy r | Mean &#124;r&#124; |
|-----:|-----:|----------:|------------:|----------:|----------:|----------:|---------:|
| 1 | H30 | +0.517 | +0.408 | +0.353 | +0.489 | +0.423 | 0.438 |
| 2 | H4 | +0.608 | -0.321 | +0.570 | -0.190 | -0.008 | 0.340 |
| 3 | H17 | +0.386 | -0.041 | +0.415 | +0.576 | +0.080 | 0.300 |
| 4 | H10 | +0.246 | +0.047 | +0.472 | +0.255 | +0.416 | 0.287 |
| 5 | H25 | -0.301 | -0.582 | -0.337 | +0.071 | +0.076 | 0.273 |
| 6 | H9 | -0.382 | -0.037 | +0.375 | +0.231 | -0.264 | 0.258 |
| 7 | H0 | -0.177 | +0.143 | -0.297 | -0.370 | -0.206 | 0.238 |
| 8 | H29 | +0.010 | +0.103 | -0.057 | +0.525 | -0.467 | 0.232 |
| 9 | H1 | -0.156 | -0.353 | -0.036 | -0.245 | -0.367 | 0.231 |
| 10 | H18 | +0.457 | +0.147 | +0.112 | -0.084 | +0.350 | 0.230 |

### Layer 18

| Rank | Head | FreeExp r | EqualProt r | DueProc r | Federal r | Privacy r | Mean &#124;r&#124; |
|-----:|-----:|----------:|------------:|----------:|----------:|----------:|---------:|
| 1 | H9 | +0.636 | +0.120 | +0.422 | +0.855 | +0.221 | 0.451 |
| 2 | H28 | +0.594 | +0.441 | +0.246 | +0.409 | +0.083 | 0.355 |
| 3 | H31 | +0.502 | +0.484 | -0.053 | +0.210 | -0.424 | 0.335 |
| 4 | H23 | -0.096 | -0.509 | -0.099 | -0.638 | -0.241 | 0.317 |
| 5 | H11 | -0.167 | -0.455 | -0.118 | +0.442 | -0.320 | 0.301 |
| 6 | H6 | +0.279 | -0.078 | -0.278 | +0.399 | +0.467 | 0.300 |
| 7 | H0 | -0.262 | +0.540 | +0.255 | -0.367 | +0.043 | 0.293 |
| 8 | H15 | +0.244 | +0.360 | +0.401 | -0.077 | +0.351 | 0.287 |
| 9 | H18 | +0.131 | +0.406 | +0.334 | -0.101 | +0.413 | 0.277 |
| 10 | H27 | +0.151 | +0.293 | -0.063 | +0.574 | +0.269 | 0.270 |

### Layer 20

| Rank | Head | FreeExp r | EqualProt r | DueProc r | Federal r | Privacy r | Mean &#124;r&#124; |
|-----:|-----:|----------:|------------:|----------:|----------:|----------:|---------:|
| 1 | H15 | +0.712 | +0.595 | +0.458 | +0.656 | +0.214 | 0.527 |
| 2 | H3 | +0.117 | +0.492 | -0.438 | +0.312 | +0.339 | 0.340 |
| 3 | H8 | +0.436 | -0.007 | -0.189 | +0.548 | -0.428 | 0.322 |
| 4 | H30 | +0.326 | +0.270 | +0.144 | +0.233 | +0.574 | 0.309 |
| 5 | H10 | +0.566 | -0.033 | -0.167 | -0.316 | +0.453 | 0.307 |
| 6 | H6 | -0.278 | +0.226 | -0.186 | -0.385 | +0.446 | 0.304 |
| 7 | H0 | +0.149 | +0.285 | +0.039 | +0.600 | +0.393 | 0.293 |
| 8 | H9 | +0.561 | +0.176 | +0.146 | +0.514 | -0.006 | 0.281 |
| 9 | H16 | -0.078 | +0.195 | +0.456 | -0.212 | -0.313 | 0.251 |
| 10 | H11 | -0.499 | -0.012 | +0.054 | -0.505 | +0.163 | 0.247 |

### Layer 21

| Rank | Head | FreeExp r | EqualProt r | DueProc r | Federal r | Privacy r | Mean &#124;r&#124; |
|-----:|-----:|----------:|------------:|----------:|----------:|----------:|---------:|
| 1 | H28 | +0.658 | +0.443 | +0.332 | +0.482 | +0.456 | 0.474 |
| 2 | H16 | +0.499 | +0.369 | +0.076 | +0.477 | +0.437 | 0.372 |
| 3 | H14 | -0.487 | -0.105 | -0.237 | -0.535 | -0.237 | 0.320 |
| 4 | H2 | +0.635 | +0.121 | +0.025 | +0.361 | +0.392 | 0.307 |
| 5 | H29 | +0.440 | +0.016 | -0.024 | +0.579 | -0.356 | 0.283 |
| 6 | H0 | -0.400 | -0.096 | +0.204 | -0.409 | -0.291 | 0.280 |
| 7 | H13 | -0.249 | +0.308 | +0.270 | -0.367 | -0.197 | 0.278 |
| 8 | H15 | +0.353 | -0.021 | +0.200 | +0.482 | +0.276 | 0.266 |
| 9 | H24 | +0.530 | +0.346 | +0.046 | -0.009 | +0.350 | 0.256 |
| 10 | H4 | +0.221 | -0.123 | -0.198 | -0.280 | -0.412 | 0.247 |

### Layer 22

| Rank | Head | FreeExp r | EqualProt r | DueProc r | Federal r | Privacy r | Mean &#124;r&#124; |
|-----:|-----:|----------:|------------:|----------:|----------:|----------:|---------:|
| 1 | H26 | +0.349 | +0.302 | +0.673 | +0.569 | +0.770 | 0.533 |
| 2 | H25 | +0.807 | +0.118 | +0.610 | +0.425 | +0.636 | 0.519 |
| 3 | H19 | +0.559 | +0.428 | +0.493 | +0.213 | +0.654 | 0.469 |
| 4 | H27 | +0.567 | +0.383 | +0.446 | +0.486 | +0.428 | 0.462 |
| 5 | H17 | +0.472 | +0.624 | +0.207 | +0.380 | +0.537 | 0.444 |
| 6 | H8 | +0.489 | +0.404 | +0.356 | +0.574 | +0.350 | 0.435 |
| 7 | H31 | +0.517 | +0.410 | +0.443 | +0.593 | -0.063 | 0.405 |
| 8 | H20 | +0.428 | +0.603 | +0.345 | +0.032 | -0.454 | 0.372 |
| 9 | H0 | +0.426 | -0.191 | +0.601 | +0.260 | +0.154 | 0.326 |
| 10 | H5 | +0.550 | +0.315 | +0.279 | +0.221 | +0.246 | 0.322 |

### Top Specialist Heads (Cross-Layer)

| Rank | Layer | Head | Top Principle | Mean &#124;r&#124; |
|-----:|------:|-----:|--------------|--------:|
| 1 | 22 | 26 | privacy_liberty | 0.533 |
| 2 | 20 | 15 | free_expression | 0.527 |
| 3 | 22 | 25 | free_expression | 0.519 |
| 4 | 21 | 28 | free_expression | 0.474 |
| 5 | 22 | 19 | privacy_liberty | 0.469 |
| 6 | 22 | 27 | free_expression | 0.462 |
| 7 | 18 | 9 | federalism | 0.451 |
| 8 | 22 | 17 | equal_protection | 0.444 |
| 9 | 17 | 30 | free_expression | 0.438 |
| 10 | 22 | 8 | federalism | 0.435 |
| 11 | 22 | 31 | federalism | 0.405 |
| 12 | 22 | 20 | equal_protection | 0.372 |
| 13 | 21 | 16 | free_expression | 0.372 |
| 14 | 18 | 28 | free_expression | 0.355 |
| 15 | 20 | 3 | equal_protection | 0.340 |
| 16 | 17 | 4 | free_expression | 0.340 |
| 17 | 18 | 31 | free_expression | 0.335 |
| 18 | 22 | 0 | due_process | 0.326 |
| 19 | 22 | 5 | free_expression | 0.322 |
| 20 | 20 | 8 | federalism | 0.322 |

## Phase 3: Attention Pattern Analysis

What do specialist heads attend to?

### L22H26 (top principle: privacy_liberty)

| Category | Mean Attention |
|----------|-------------:|
| other | 0.6303 |
| legal_term | 0.3050 |
| punctuation | 0.0587 |
| whitespace | 0.0060 |
| principle:free_expression | 0.0000 |
| principle:due_process | 0.0000 |
| principle:equal_protection | 0.0000 |
| principle:privacy_liberty | 0.0000 |
| principle:federalism | 0.0000 |

### L20H15 (top principle: free_expression)

| Category | Mean Attention |
|----------|-------------:|
| legal_term | 0.4967 |
| other | 0.4067 |
| punctuation | 0.0746 |
| whitespace | 0.0212 |
| principle:due_process | 0.0005 |
| principle:privacy_liberty | 0.0001 |
| principle:free_expression | 0.0001 |
| principle:equal_protection | 0.0001 |
| principle:federalism | 0.0000 |

### L22H25 (top principle: free_expression)

| Category | Mean Attention |
|----------|-------------:|
| other | 0.5251 |
| punctuation | 0.2186 |
| legal_term | 0.1166 |
| whitespace | 0.0930 |
| principle:free_expression | 0.0308 |
| principle:privacy_liberty | 0.0083 |
| principle:due_process | 0.0036 |
| principle:equal_protection | 0.0025 |
| principle:federalism | 0.0015 |

### L21H28 (top principle: free_expression)

| Category | Mean Attention |
|----------|-------------:|
| other | 0.6425 |
| whitespace | 0.1942 |
| punctuation | 0.0836 |
| legal_term | 0.0793 |
| principle:free_expression | 0.0002 |
| principle:due_process | 0.0001 |
| principle:equal_protection | 0.0000 |
| principle:privacy_liberty | 0.0000 |
| principle:federalism | 0.0000 |

### L22H19 (top principle: privacy_liberty)

| Category | Mean Attention |
|----------|-------------:|
| other | 0.5581 |
| legal_term | 0.4208 |
| punctuation | 0.0168 |
| whitespace | 0.0037 |
| principle:privacy_liberty | 0.0004 |
| principle:due_process | 0.0002 |
| principle:federalism | 0.0001 |
| principle:free_expression | 0.0000 |
| principle:equal_protection | 0.0000 |

### L22H27 (top principle: free_expression)

| Category | Mean Attention |
|----------|-------------:|
| other | 0.7064 |
| punctuation | 0.1217 |
| legal_term | 0.0873 |
| whitespace | 0.0820 |
| principle:free_expression | 0.0013 |
| principle:equal_protection | 0.0006 |
| principle:due_process | 0.0006 |
| principle:federalism | 0.0001 |
| principle:privacy_liberty | 0.0001 |

### L18H9 (top principle: federalism)

| Category | Mean Attention |
|----------|-------------:|
| legal_term | 0.5771 |
| other | 0.3358 |
| punctuation | 0.0568 |
| whitespace | 0.0251 |
| principle:due_process | 0.0020 |
| principle:equal_protection | 0.0016 |
| principle:free_expression | 0.0009 |
| principle:privacy_liberty | 0.0004 |
| principle:federalism | 0.0004 |

### L22H17 (top principle: equal_protection)

| Category | Mean Attention |
|----------|-------------:|
| other | 0.7217 |
| punctuation | 0.1439 |
| legal_term | 0.0668 |
| whitespace | 0.0665 |
| principle:free_expression | 0.0006 |
| principle:equal_protection | 0.0003 |
| principle:federalism | 0.0001 |
| principle:due_process | 0.0001 |
| principle:privacy_liberty | 0.0000 |

### L17H30 (top principle: free_expression)

| Category | Mean Attention |
|----------|-------------:|
| punctuation | 0.3968 |
| other | 0.3599 |
| legal_term | 0.1223 |
| whitespace | 0.1193 |
| principle:free_expression | 0.0009 |
| principle:due_process | 0.0007 |
| principle:federalism | 0.0001 |
| principle:equal_protection | 0.0000 |
| principle:privacy_liberty | 0.0000 |

### L22H8 (top principle: federalism)

| Category | Mean Attention |
|----------|-------------:|
| whitespace | 0.4733 |
| other | 0.3958 |
| punctuation | 0.1181 |
| legal_term | 0.0122 |
| principle:due_process | 0.0002 |
| principle:free_expression | 0.0001 |
| principle:equal_protection | 0.0001 |
| principle:federalism | 0.0001 |
| principle:privacy_liberty | 0.0000 |

## Interpretation

Top discriminative components: attn_22 (mean &#124;r&#124;=0.882), attn_21 (0.865), attn_20 (0.779)

In top-10 discriminative components: 8 attention, 2 MLP, 0 other

# RCI: Recursive Convergent Inference

### Bottom-Up Module Expansion via Output Convergence

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKGM_matDR-JZb-qk20x2gQ3aYiovhqx?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Preprint%20Ready-green)](https://github.com/olanokhin/rci-inference)
[![Author](https://img.shields.io/badge/Author-Oleksandr_Anokhin-purple)](https://olanokhin.com)

> *Stop when the model agrees with itself — not when you tell it to.*

---

## Abstract

We propose **Recursive Convergent Inference (RCI)**, an architectural principle for neural network inference in which the set of active computational modules expands monotonically from a minimal seed subset until empirical convergence of the model's next-token output distribution.

Unlike existing adaptive computation methods that determine *when to halt* a fixed computation, RCI determines *when additional computation is warranted* — growing the active module set via breadth-first search over a precomputed affinity graph until output stability is reached. Stopping requires no learned halting signal, external verifier, or task-complexity pre-estimator.

RCI shifts from **external scaling** (longer outputs, multiple samples, verifier-guided search) to **internal scaling** over the active parameter subgraph.

---

## Key Results

Evaluated on **OLMoE-1B-7B** (64 experts) across **n=150** reasoning tasks (50 per difficulty tier):

| Difficulty | Benchmark | n | Avg AUC | Std |
|:---|:---|:---:|:---:|:---:|
| Easy | GSM8K | 50 | 10.728 | 2.808 |
| Medium | MATH (algebra) | 50 | 8.956 | 1.688 |
| Hard | MMLU hard subsets | 50 | 11.987 | 2.537 |

**Statistical significance (n=150):**
- Hard vs Easy: Mann-Whitney U=1677, **p=0.002**
- Hard vs Medium: U=2106, **p<0.001**
- Easy vs Medium: U=1788, **p<0.001**
- Spearman ρ=0.22, **p=0.007**, n=150

**Notable finding:** RCI's complexity metric diverges from human-defined difficulty labels — MATH algebra is treated as computationally simpler than GSM8K word problems by this model, suggesting RCI captures **model-relative computational demand** rather than task difficulty in the abstract.

---

## How It Works

```
Weights W (read-only, shared)
         │
    M₀ = seeds (top activated experts on first pass)
         │
    Step n:  Mₙ₊₁ = Mₙ ∪ top-k(neighbors(Mₙ), affinity)
         │
    Stop when: rolling KL(probsₙ || probsₙ₋₁) < ε
               AND confidence margin ≥ θ
         │
    Result: easy task  → few experts, few steps
            hard task  → more experts, more steps
            automatically, without external signal
```

---

## Reproducibility

All experiments reproducible on **free-tier Google Colab T4 GPU** (~60 minutes).

**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKGM_matDR-JZb-qk20x2gQ3aYiovhqx?usp=sharing)**

**Setup:**
1. Open notebook in Colab
2. Add `HF_TOKEN` to Colab Secrets (left panel → 🔑)
3. Run Cell 1 → Restart runtime → Run all

**Model:** [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924) — fully open, Apache 2.0

---

## Repository Structure

```
rci-inference/
├── rci_inference_poc.ipynb   # Full experiment notebook
├── rci_results.json          # Experimental results
├── rci_figure1.png           # Results figure
└── README.md
```

---

## Citation

```bibtex
@misc{anokhin2026rci,
  title  = {Recursive Convergent Inference: Bottom-Up Module
             Expansion via Output Convergence},
  author = {Anokhin, Alex},
  year   = {2026},
  note   = {Preprint. github.com/olanokhin/rci-inference}
}
```

---

**Author:** Alex Anokhin · [olanokhin@gmail.com](mailto:olanokhin@gmail.com) · [LinkedIn](https://linkedin.com/in/olanokhin)
**Date:** March 2026

# RCI: Recursive Convergent Inference

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Paper%20%2B%20Colab%20Ready-green.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/[YOUR_NOTEBOOK_LINK])

> *Stop when the model agrees with itself — not when you tell it to.*

---

## Abstract

We propose **Recursive Convergent Inference (RCI)**, an architectural principle for neural network inference in which the set of active computational modules expands monotonically from a minimal seed subset until empirical convergence of the model's next-token output distribution.

Unlike existing adaptive computation methods that determine *when to halt* a fixed computation, RCI determines *when additional computation is warranted* — progressively incorporating additional modules based on inter-module affinities until output stability is reached.

**Key properties:**
- No learned halting signal required
- No external verifier or task-complexity pre-estimator
- Stopping emerges directly from internal output distribution stability
- Naturally enables parallel multi-task execution over shared read-only weights

RCI shifts from **external scaling** (longer outputs, multiple samples, verifier-guided search) to **internal scaling** over the active parameter subgraph.

---

## Key Results

Instantiated on **OLMoE-1B-7B** (64 discrete experts, fully open model).

Evaluated on reasoning benchmarks of varying difficulty (GSM8K, MATH, MMLU hard subsets):

| Finding | Result |
|---|---|
| Hard vs Easy tasks — KL divergence | Mann-Whitney U, **p=0.007** |
| Hard vs Medium tasks — KL divergence | Mann-Whitney U, **p=0.001** |
| Difficulty vs computational effort | Spearman ρ=0.45, **p=0.013** (n=30) |

> RCI's complexity metric diverges from human-defined difficulty labels — suggesting it captures **model-relative computational demand** rather than task difficulty in the abstract.

---

## How It Works

```
Inference as monotonic expansion:

  Step 0:  seed subset S₀ ⊂ M (minimal active modules)
  Step n:  expand Sₙ based on inter-module affinities
  Stop:    when hybrid distributional stability metric < threshold
           + safeguards against premature convergence

  Result:  easy task  → few modules activated
           hard task  → more modules, more compute
           automatically, without external signal
```

---

## Why It Matters

```
Existing test-time compute scaling:
  chain-of-thought     → longer outputs
  self-consistency     → multiple samples
  verifier-guided      → external model

RCI:
  internal scaling     → active parameter subgraph
  no extra tokens      → no extra output length
  no verifier          → self-contained
  parallelizable       → shared read-only weights
```

---

## Reproducibility

Full implementation released as a Google Colab notebook.  
All experiments reproducible on free-tier GPU.

**→ [Open in Colab](#)** *(link to be added)*

Model: [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B) — fully open weights, Apache 2.0.

---

## Paper Structure

1. Introduction
2. Related Work
3. Method — Monotonic Module Expansion
4. Experiments — GSM8K, MATH, MMLU
5. Limitations & Conclusion

Full paper: `paper/rci-paper.pdf` *(coming soon)*  
arXiv preprint: *in preparation*

---

## Citation

```bibtex
@misc{anokhin2026rci,
  title  = {Recursive Convergent Inference},
  author = {Anokhin, Alex},
  year   = {2026},
  note   = {Preprint. github.com/olanokhin/rci-inference}
}
```

---

**Author:** Alex Anokhin · [olanokhin@gmail.com](mailto:olanokhin@gmail.com)  
**Date:** March 2026

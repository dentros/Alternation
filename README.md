# Alternation Measures

## Citation

If you use this repository, the data, the Honey-Jar Game (HJG), the Alternation (ALT) metrics, or the accompanying evaluation framework in your research, please cite:

Papadopoulos, N. A., & Psannis, K. E. (2026). *The Coordination Gap: Multi-Agent Alternation Metrics for Temporal Fairness in Repeated Games*. arXiv:2603.05789. https://doi.org/10.48550/arXiv.2603.05789

```bibtex
@article{papadopoulos2026coordinationgap,
  title={The Coordination Gap: Multi-Agent Alternation Metrics for Temporal Fairness in Repeated Games},
  author={Papadopoulos, Nikolaos Al. and Psannis, Konstantinos E.},
  journal={arXiv preprint arXiv:2603.05789},
  year={2026},
  doi={10.48550/arXiv.2603.05789}
}
```

Code and data archive (Zenodo): https://doi.org/10.5281/zenodo.18528891

## Repository contents

- Python source (`*.py`): environment, agents, metrics, experiments, and visualization code.
- `checkpoints/`: experiment result files (BASE=1000, Q-learning), one `.pkl` per configuration (`result_{n}agents_Type-{A,B}_{ILF,IQF}.pkl`) and one per random-policy baseline (`random_{n}agents_{ILF,IQF}.pkl`), for `n ∈ {2,3,5,8,10}`.
- `figures/main/`: the five main-text figures.
- `figures/supplementary/appendix/`: supplementary figures, including benchmarking curves, learning curves, Q-table heatmaps, and PA-equivalent comparisons.

## Data availability

Two checkpoint files (`result_8agents_Type-B_ILF.pkl` and `result_8agents_Type-B_IQF.pkl`, approximately 235 MB each) exceed GitHub's file size limit and are therefore not hosted in this repository. They are available from the corresponding author upon reasonable request (nikolaos.papadopoulos@uom.edu.gr).

## Ownership

The original implementation of the Honey-Jar Game (HJG), the Alternation (ALT) metrics, the evaluation framework, and all source code in this repository were developed by Nikolaos Al. Papadopoulos.

Copyright (c) 2026 Nikolaos Al. Papadopoulos.

Distributed under the GNU General Public License v3.0.

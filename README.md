# Alternation Measures

## Citation

If you use this repository, the data, the Honey-Jar Game (HJG), the Alternation (ALT) metrics, or the accompanying evaluation framework in your research, please cite:

Papadopoulos, N. A., & Psannis, K. E. (2026). *The Coordination Gap: Multi-Agent Alternation Metrics for Temporal Fairness in Repeated Games*. arXiv:2603.05789. https://doi.org/10.48550/arXiv.2603.05789

@article{papadopoulos2026coordinationgap,
  title={The Coordination Gap: Multi-Agent Alternation Metrics for Temporal Fairness in Repeated Games},
  author={Papadopoulos, Nikolaos Al. and Psannis, Konstantinos E.},
  journal={arXiv preprint arXiv:2603.05789},
  year={2026},
  doi={10.48550/arXiv.2603.05789}


Code & Data Archived version (Zenodo): https://doi.org/10.5281/zenodo.18528891


## Repository contents

- Python source (`*.py`): environment, agents, metrics, experiments, and visualization code.
- `checkpoints/`: experiment result files (BASE=1000, Q-learning), one `.pkl` per configuration (`result_{n}agents_Type-{A,B}_{ILF,IQF}.pkl`) and one per random-policy baseline (`random_{n}agents_{ILF,IQF}.pkl`), for n in {2, 3, 5, 8, 10}.
- `figures/main/`: the five main-text figures.
- `figures/supplementary/appendix/`: the supplementary figures (benchmarking curves, learning curves, Q-table heatmaps, PA-equivalent comparisons).

## Data availability note

Two checkpoint files (`result_8agents_Type-B_ILF.pkl` and `result_8agents_Type-B_IQF.pkl`, approximately 235 MB each) exceed GitHub's file size limit and are not hosted in this repository. They are available on request from the corresponding author (nikolaos.papadopoulos@uom.edu.gr).

## Ownership

Nikolaos Al. Papadopoulos


## License

GNU GPL

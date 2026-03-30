# SoftMol vs. Beyond Affinity Benchmark on 10 Protein Targets

## Benchmark Citation

Zheng, Kangyu, Kai Zhang, Jiale Tan, Xuehan Chen, Yingzhou Lu, Zaixi Zhang, Lichao Sun, Marinka Zitnik, Tianfan Fu, and Zhiding Liang. “Beyond Affinity: A Benchmark of 1D, 2D, and 3D Methods Reveals Critical Trade-offs in Structure-Based Drug Design.” *Transactions on Machine Learning Research*, 2026.

Protocol: 15 benchmark methods, 10 targets, 1,000 molecules per target, lower docking score is better.

## Summary

SoftMol achieves state-of-the-art performance on all 10 targets in this benchmark. For Top-1, Top-10, and Top-100 docking score, the best reported baseline is surpassed on every target, and the strongest average baseline, Pocket2Mol, is improved from `-13.24 / -12.66 / -11.63` to `-15.22 / -14.46 / -13.44`, respectively. These results indicate that the gain is not concentrated on a small subset of targets, but remains consistent across the full 10-target benchmark.

## Average Summary

| Method | Top-1 | Top-10 | Top-100 |
| --- | ---: | ---: | ---: |
| Pocket2Mol | -13.24 | -12.66 | -11.63 |
| JT-VAE | -11.81 | -10.70 | -9.38 |
| SMILES-LSTM-HC | -11.80 | -11.00 | -10.03 |
| SMILES-VAE | -11.39 | -10.65 | -9.65 |
| PocketFlow | -11.37 | -10.79 | -9.40 |
| REINVENT | -11.13 | -10.52 | -9.58 |
| 3DSBDD | -11.08 | -10.61 | -9.44 |
| Pasithea | -11.05 | -10.56 | -9.67 |
| DST | -11.03 | -10.20 | -9.22 |
| MIMOSA | -10.97 | -10.26 | -9.23 |
| SMILES-GA | -10.74 | -10.09 | -8.86 |
| graph-GA | -10.50 | -9.70 | -8.17 |
| ResGen | -9.28 | -9.01 | -8.25 |
| TargetDiff | -8.86 | -9.14 | -9.84 |
| MoLDQN | -7.71 | -7.04 | -5.79 |
| **SoftMol (Unconstrained)** | **-15.22** | **-14.46** | **-13.44** |

## Top-1 Docking Score

Best value in each target column is highlighted in bold.

| Method | 6GL8 | 1UWH | 7OTE | 1KKQ | 5WFD | 7W7C | 8JJL | 7D42 | 7S1S | 6AZV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pocket2Mol | -11.56 | -14.56 | -15.72 | -14.18 | -11.30 | -13.76 | -13.27 | -12.76 | -12.90 | -12.36 |
| JT-VAE | -10.26 | -12.38 | -12.29 | -12.25 | -11.65 | -12.45 | -11.91 | -12.79 | -11.53 | -10.60 |
| SMILES-LSTM-HC | -9.77 | -11.50 | -12.35 | -12.40 | -11.21 | -13.93 | -11.41 | -12.84 | -12.02 | -10.61 |
| SMILES-VAE | -9.35 | -11.95 | -13.06 | -10.91 | -10.01 | -12.11 | -11.95 | -12.01 | -12.14 | -10.42 |
| PocketFlow | -9.42 | -11.04 | -10.27 | -12.47 | -10.30 | -13.52 | -11.88 | -12.79 | -10.16 | -11.83 |
| REINVENT | -9.06 | -11.13 | -12.03 | -11.19 | -10.18 | -11.88 | -11.63 | -12.23 | -11.32 | -10.66 |
| 3DSBDD | -8.61 | -12.67 | -10.52 | -13.36 | -11.28 | -11.29 | -11.67 | -11.12 | -10.19 | -10.13 |
| Pasithea | -9.25 | -11.47 | -11.56 | -10.45 | -10.54 | -12.00 | -11.87 | -11.76 | -11.35 | -10.24 |
| DST | -8.69 | -11.09 | -11.41 | -10.92 | -10.13 | -12.14 | -12.20 | -11.87 | -11.54 | -10.31 |
| MIMOSA | -8.64 | -11.13 | -11.49 | -11.00 | -9.91 | -11.72 | -11.85 | -11.72 | -11.96 | -10.27 |
| SMILES-GA | -8.83 | -10.74 | -11.18 | -10.47 | -9.72 | -11.74 | -11.29 | -11.93 | -11.05 | -10.46 |
| graph-GA | -8.47 | -11.19 | -11.15 | -10.61 | -9.66 | -11.85 | -10.72 | -11.03 | -10.47 | -9.85 |
| ResGen | – | -9.71 | -7.14 | -9.81 | – | -7.88 | – | -8.94 | -11.77 | -9.71 |
| TargetDiff | – | -7.30 | – | -11.53 | -10.07 | -9.27 | -8.91 | -6.10 | – | – |
| MoLDQN | -6.63 | -7.38 | -7.49 | -7.35 | -7.79 | -7.75 | -8.94 | -7.30 | -8.42 | -8.04 |
| **SoftMol (Unconstrained)** | **-13.20** | **-15.00** | **-16.30** | **-15.30** | **-14.50** | **-16.80** | **-14.70** | **-15.50** | **-16.10** | **-14.80** |

## Top-10 Docking Score

Best value in each target column is highlighted in bold.

| Method | 6GL8 | 1UWH | 7OTE | 1KKQ | 5WFD | 7W7C | 8JJL | 7D42 | 7S1S | 6AZV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pocket2Mol | -10.53 | -13.87 | -14.78 | -13.53 | -11.06 | -13.56 | -12.79 | -12.10 | -12.49 | -11.87 |
| JT-VAE | -9.09 | -11.12 | -11.27 | -10.90 | -10.26 | -10.81 | -10.81 | -11.52 | -11.04 | -10.18 |
| SMILES-LSTM-HC | -9.26 | -11.02 | -11.54 | -11.21 | -10.41 | -12.20 | -11.09 | -11.47 | -11.47 | -10.38 |
| SMILES-VAE | -8.76 | -11.13 | -11.35 | -10.31 | -9.74 | -11.42 | -11.14 | -11.42 | -11.14 | -10.11 |
| PocketFlow | -8.98 | -10.68 | -9.55 | -11.67 | -9.34 | -13.02 | -11.42 | -12.25 | -9.46 | -11.49 |
| REINVENT | -8.65 | -10.72 | -10.76 | -10.51 | -9.56 | -11.52 | -11.13 | -11.40 | -10.99 | -9.92 |
| 3DSBDD | -8.19 | -12.41 | -10.41 | -12.71 | -10.72 | -10.78 | -10.90 | -10.43 | -9.84 | -9.66 |
| Pasithea | -8.73 | -11.12 | -10.84 | -10.13 | -9.88 | -11.41 | -11.24 | -11.15 | -11.06 | -10.03 |
| DST | -8.23 | -10.48 | -10.76 | -9.96 | -9.22 | -11.11 | -10.79 | -10.92 | -10.73 | -9.78 |
| MIMOSA | -8.28 | -10.52 | -10.82 | -10.16 | -9.35 | -11.14 | -10.77 | -11.03 | -10.78 | -9.78 |
| SMILES-GA | -8.44 | -10.33 | -10.27 | -9.81 | -9.20 | -10.97 | -11.01 | -11.01 | -10.46 | -9.41 |
| graph-GA | -7.94 | -10.12 | -9.94 | -9.63 | -9.03 | -10.60 | -10.23 | -10.43 | -9.98 | -9.14 |
| ResGen | – | -9.23 | -7.03 | -9.56 | – | -7.56 | – | -8.60 | -11.63 | -9.47 |
| TargetDiff | – | – | – | -11.14 | -8.29 | – | -7.99 | – | – | – |
| MoLDQN | -6.13 | -6.69 | -6.93 | -6.75 | -7.37 | -7.33 | -7.90 | -6.53 | -8.01 | -6.73 |
| **SoftMol (Unconstrained)** | **-12.17** | **-14.38** | **-15.33** | **-14.64** | **-13.86** | **-16.17** | **-13.98** | **-14.94** | **-14.87** | **-14.25** |

## Top-100 Docking Score

Best value in each target column is highlighted in bold.

| Method | 6GL8 | 1UWH | 7OTE | 1KKQ | 5WFD | 7W7C | 8JJL | 7D42 | 7S1S | 6AZV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pocket2Mol | -9.90 | -12.81 | -13.14 | -12.36 | -10.45 | -12.55 | -11.63 | -10.88 | -11.51 | -11.11 |
| JT-VAE | -7.93 | -9.82 | -9.65 | -9.62 | -9.00 | -8.63 | -9.78 | -10.30 | -10.07 | -9.02 |
| SMILES-LSTM-HC | -8.29 | -10.18 | -10.40 | -10.10 | -9.43 | -11.09 | -10.16 | -10.62 | -10.51 | -9.52 |
| SMILES-VAE | -7.97 | -9.97 | -10.09 | -9.08 | -8.84 | -10.54 | -10.17 | -10.34 | -10.22 | -9.32 |
| PocketFlow | -7.92 | -8.12 | -7.74 | -10.33 | -8.14 | -11.94 | -10.35 | -11.11 | -7.82 | -10.56 |
| REINVENT | -7.82 | -9.70 | -9.77 | -9.40 | -8.80 | -10.46 | -10.26 | -10.32 | -10.08 | -9.19 |
| 3DSBDD | -7.22 | -11.52 | -9.67 | -11.59 | -9.86 | -8.67 | -9.58 | -9.10 | -8.95 | -8.28 |
| Pasithea | -7.93 | -9.94 | -9.98 | -9.36 | -8.92 | -10.53 | -10.22 | -10.31 | -10.23 | -9.30 |
| DST | -7.50 | -9.41 | -9.47 | -9.00 | -8.41 | -10.06 | -9.74 | -9.92 | -9.78 | -8.90 |
| MIMOSA | -7.49 | -9.54 | -9.49 | -8.99 | -8.46 | -10.08 | -9.74 | -9.88 | -9.77 | -8.85 |
| SMILES-GA | -7.37 | -8.94 | -8.95 | -8.64 | -8.12 | -9.78 | -9.61 | -9.39 | -9.30 | -8.50 |
| graph-GA | -6.81 | -8.32 | -8.25 | -8.07 | -7.65 | -8.89 | -8.52 | -8.82 | -8.46 | -7.88 |
| ResGen | – | -8.06 | -6.30 | -8.82 | – | -7.04 | – | -7.71 | -11.06 | -8.75 |
| TargetDiff | – | – | – | -9.84 | – | – | – | – | – | – |
| MoLDQN | -4.84 | -5.46 | -5.87 | -5.65 | -5.79 | -6.12 | -6.57 | -5.41 | -6.38 | -5.78 |
| **SoftMol (Unconstrained)** | **-11.30** | **-13.72** | **-14.12** | **-13.67** | **-12.63** | **-14.78** | **-13.11** | **-14.23** | **-13.89** | **-12.98** |

## Ready-to-Paste BibTeX

```bibtex
@article{zheng2026beyond_affinity,
  title   = {Beyond Affinity: A Benchmark of 1D, 2D, and 3D Methods Reveals Critical Trade-offs in Structure-Based Drug Design},
  author  = {Zheng, Kangyu and Zhang, Kai and Tan, Jiale and Chen, Xuehan and Lu, Yingzhou and Zhang, Zaixi and Sun, Lichao and Zitnik, Marinka and Fu, Tianfan and Liang, Zhiding},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=gaTwx1rzCw}
}
```

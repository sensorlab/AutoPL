# AutoPathloss: Pathloss Approximation with Kolmogorov-Arnold Networks

## Overview

This repository contains all the scripts necessary to reproduce the experiments corresponding to the results reported in the AutoPL paper. Everything was run with python 3.11. 

For the KAN-based AutoPL, a suite of Jupyter notebooks implementing automatic pathloss approximation using Kolmogorov-Arnold (KA) networks are provided. KAN networks are a class of neural networks particularly suited for function approximation. We explore their performance on both measured and synthetic pathloss data.

For the DSO, three scripts are provided. One runs the experiment, one aggregates the results and one plots figures. As the original Deep Symbolic optimization Github library is not maintained anymore, we made a fork and upgraded it to enable running on python 3.11 https://github.com/cfortuna/deep-symbolic-optimization. 

We study the feasibility of approximation using data generated with the existing ABG and CI models as well as with empirical measurements.


## Cite
If you use this code in your research, citation of the following paper would be greatly appreciated:
```
    @inproceedings{
        anaqreh2024towards,
        title={Towards Automated and Interpretable Pathloss Approximation  Methods},
        author={Ahmad Anaqreh and Shih-Kai Chou and Irina Bara{\v{s}}in and Carolina Fortuna},
        booktitle={Submitted to AAAI 2025 Workshop on Artificial Intelligence for Wireless Communications and Networking (AI4WCN)},
        year={2024},
        url={https://openreview.net/forum?id=M1WT5NZ4bj}
    }
    @article{anaqreh2025automated,
      title={Automated Modeling Method for Pathloss Model Discovery},
      author={Anaqreh, Ahmad and Chou, Shih-Kai and Mohor{\v{c}}i{\v{c}}, Mihael and Fortuna, Carolina},
      journal={arXiv preprint arXiv:2505.23383},
      year={2025}
      url={https://arxiv.org/pdf/2505.23383?}
    }
```

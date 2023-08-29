# multilevelDiff
Multilevel Diffusion: Infinite dimensional score

Code accommpanying the paper "Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation" (arXiv: 2303.04772).
This repo heavily builds upon the repo https://github.com/CW-Huang/sdeflow-light from [1]. Furthermore, for the FNO [2] implementation we use an 
earlier version of the official FNO repo https://github.com/neuraloperator/neuraloperator. For the metrics we build upon the repo https://github.com/layer6ai-labs/dgm-eval from [3] 
and the Riesz MMD [4] is from the repo https://github.com/fabianaltekrueger/neuralwassersteingradientflows.

Linked papers: 
[1] A Variational Perspective on Diffusion-Based Generative Models and Score Matching, Huang et al, NeurIPS 2021
[2] Fourier Neural Operator for Parametric Partial Differential Equations, Li et al., ICLR 2021
[3] Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models, Caterini et al, arXiv 2306.04675
[4] Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels, Altekrueger et al, ICML 2023

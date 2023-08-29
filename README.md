# multilevelDiff
Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation

Code accommpanying the paper "Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation" (arXiv: 2303.04772)[1].
This repo heavily builds upon the repo https://github.com/CW-Huang/sdeflow-light from [2]. Furthermore, for the FNO [3] implementation we use an 
earlier version of the official FNO repo https://github.com/neuraloperator/neuraloperator. For the metrics we build upon the repo https://github.com/layer6ai-labs/dgm-eval from [4] 
and the Riesz MMD [5] is from the repo https://github.com/fabianaltekrueger/neuralwassersteingradientflows.

Linked papers: 
[1] Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation, Hagemann et al, arXiv 2303.04772

[2] A Variational Perspective on Diffusion-Based Generative Models and Score Matching, Huang et al, NeurIPS 2021

[3] Fourier Neural Operator for Parametric Partial Differential Equations, Li et al., ICLR 2021

[4] Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models, Caterini et al, arXiv 2306.04675

[5] Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels, Altekrueger et al, ICML 2023

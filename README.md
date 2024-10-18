# Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation

This repo includes the implementation of well-posed infinite dimensional SBDM. This code accommpanies the paper "Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation" (arXiv: 2303.04772)[1].
We explore diffusion models in function spaces for generating images at varying resolutions.


This repo heavily builds upon the repo https://github.com/CW-Huang/sdeflow-light from [2]. 
Furthermore, for the FNO [3] implementation we use an 
earlier version of the official FNO repo https://github.com/neuraloperator/neuraloperator.

For the metrics we build upon the repo https://github.com/layer6ai-labs/dgm-eval from [4] 
and the Riesz MMD [5] is from the repo https://github.com/fabianaltekrueger/neuralwassersteingradientflows.


## Linked papers: 
[1] Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation, Hagemann et al, arXiv 2303.04772

[2] A Variational Perspective on Diffusion-Based Generative Models and Score Matching, Huang et al, NeurIPS 2021

[3] Fourier Neural Operator for Parametric Partial Differential Equations, Li et al., ICLR 2021

[4] Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models, Caterini et al, arXiv 2306.04675

[5] Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels, Altekrueger et al, ICML 2023

## Usage

Train the Infinite Dimensional Score-Based Diffusion Models, and save a checkpoint
```
python main.py
```

You can use the default hyperparameters. Or you can change --dataset, --prior_name, --model to customize the generated dataset, prior, and architecture, respectively. The best hyperparameters for different setups are elaborated in the paper.
```
main.py 
    [--dataset {MNIST,FashionMNIST}] [--n_epochs N_EPOCHS] [--lr {0.001,0.0001}]
    [--batch_size {256}] [--num_samples NUM_SAMPLES] [--num_samples_mmd NUM_SAMPLES_MMD]
    [--num_steps {200}] [--input_height INPUT_HEIGHT]
    [--prior_name {fno,combined_conv,lap_conv,standard}] [--width {32,64,128}]
    [--model {unet,fno}] [--modes {12,14,15}] [--viz_freq VIZ_FREQ] [--val_freq VAL_FREQ]
    [--seed SEED] [--out_dir OUT_DIR] [--out_file OUT_FILE] [--save SAVE]
```

For PDE experiments, download the dataset from PDEBench (https://github.com/pdebench/PDEBench/tree/main). For example,
```
git clone https://github.com/pdebench/PDEBench.git
python download_direct.py --root_folder $proj_home/data --pde_name 2d_reacdiff
```
Then run
```
main_pde.py 
    [--prior_name {bessel,fno,combined_conv,lap_conv,standard}]
```


Generate new samples and the corresponding MMD, diversity metrics
```
python test.py
    --save_model [add your saved checkpoint here].pt
    --dataset [dataset used in the checkpoint. MNIST or FashionMNIST]
```

## Quick Overview
- `fno.py` Fourier neural operator layer and network used in the architecture, and the prior.
- `sde.py` Forward and reverse SDEs, denoising score matching loss.
- `metrics.py` MMD and Vendi diversity score.


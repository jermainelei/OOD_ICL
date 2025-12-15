# OOD_ICL — In-Context Learning with Training on Low-dimensional Subspaces  
Code for **“In-Context Learning with Training on Low-dimensional Subspaces for Out of Distribution Tasks”**    
Mishty Dhekial, Jermaine Lei, Imra Dawoodani, David Yang (UC Berkeley)  
This repository studies *when* transformer in-context learning (ICL) generalizes **out-of-distribution (OOD)** by controlling the **geometry and dimensional structure** of the *task distribution* during pretraining.  
We build on the linear-regression ICL framework popularized by Goddard et al. (ICML 2025) and extend it to **union-of-subspaces** task distributions (low intrinsic dimension, multiple subspaces, controllable directional coverage).  
> **Upstream / credit:** This project uses and modifies the GPT-2 style regression-transformer training/eval pipeline from Chase Goddard et al.’s codebase:    
> https://github.com/cwgoddard/OOD_ICL  
---  
## What problem does this code implement?  
### In-context linear regression (task family)  
Each task is a noiseless linear regression problem:  
- sample inputs: x ~ N(0, I_d)  
- labels: y = w^T x 
- in a context of n-1 labeled examples plus a final query `x_n`, the transformer predicts `ŷ_n`.  
### Training task distribution: union of low-dimensional subspaces  
We pretrain on task weight vectors `w ∈ R^d` that **do not** fill the whole ambient space uniformly.  
Instead, we:  
1. sample **c** random **k-dimensional** subspaces (with `k << d`),  
2. sample intrinsic directions `u` inside each subspace from a spherical-cap distribution controlled by **ϕ** (cap half-angle),  
3. embed to the ambient space with `w = Q(m) u` and normalize.  
Intuition:  
- **k** controls *intrinsic dimension* (how low-rank each subspace is),  
- **c** controls *how many distinct subspaces* we cover during pretraining,  
- **ϕ** controls *within-subspace directional diversity*.  
---  
## Evaluation: In-distribution vs Out-of-distribution  
We evaluate on two complementary test distributions:  
### In-distribution (ID)  
Tasks are sampled from the **same union of training subspaces** (reuse the same subspace bases and resample intrinsic directions).  
### Out-of-distribution (OOD)  
Tasks are sampled from the **full ambient hypersphere** in `R^d\` (no restriction to the training subspaces).  
### Cone falloff (OOD vs angle)
Following Goddard et al., we also evaluate OOD performance as a function of **angular distance δ** from the training region by sampling test weights from spherical bands at increasing angles.  
---  
## Repository layout  
- `experiments.ipynb` — main entry point for running training + sweeps + plotting  
- `src/` — implementation utilities used by the notebook (model, sampling, training, eval)  
- `fig_sphere.png` — schematic figure used in documentation/plots  

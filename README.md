# Fourier Neural Operator (FNO) for 2D Darcy Flow

## Overview
This repository contains the implementation of a Fourier Neural Operator (FNO) designed to solve parametric Partial Differential Equations (PDEs), specifically the 2D steady-state Darcy Flow equation. 

Unlike standard Convolutional Neural Networks (CNNs) that learn mappings between finite-dimensional grids, this FNO learns the continuous integral operator itself. This renders the architecture inherently **mesh-invariant** and capable of zero-shot super-resolution.

*This project was completed as an Undergraduate Research Project at IIT (BHU), Varanasi.*

## Mathematical Architecture
The model replaces the discrete weight matrix of classical neural networks with a learnable continuous kernel function parameterized in the Fourier domain.
1. **Lifting:** Input permeability field $a(x)$ is lifted to a higher-dimensional channel space.
2. **Fourier Layers:** The core integral operator is computed using the Fast Fourier Transform (FFT). By truncating high-frequency modes and applying the Convolution Theorem, the $O(n^2)$ integration complexity is reduced to $O(n \log n)$.
3. **Projection:** The hidden representation is projected back to the target dimension to output the pressure field $u(x)$.

## Performance & Zero-Shot Super-Resolution
The model was trained exclusively on a coarse $32 \times 32$ spatial grid. 
To validate mesh-invariance, the frozen model was evaluated on a $64 \times 64$ grid without any retraining or architectural modification. 

* **Training:** 100 Epochs (Adam Optimizer). Absolute error residuals converged to the $10^{-4} - 10^{-6}$ range.
* **Super-Resolution Accuracy:** Achieved a relative $L^2$ error of **0.0822** on the unseen $64 \times 64$ grid, accurately reproducing the high-fidelity pressure distribution while achieving orders-of-magnitude speedup over classical Finite Difference numerical solvers.

## Project Structure
* `model.py`: Contains the PyTorch implementation of the Spectral Convolution and FNO architecture.
* `train.py`: Training loop and loss computation.

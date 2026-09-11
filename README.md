# Lightweight PDE Model For Poisson Denoising

Welcome to the official repository for my research paper and project, **"Lightweight PDE Model For Poisson Denoising"**, co-authored with L Joy Nirantar. We developed this mathematically transparent, "white-box" framework to address the limitations of heavy, unconstrained convolutional networks in image restoration. This repository contains the complete PyTorch implementation of our work.

## Overview
Standard image restoration models often struggle in low-photon environments, such as fluorescence microscopy and astronomical imaging, where noise adheres to a signal-dependent Poisson probability distribution. While deep convolutional frameworks like TRDPD offer high performance, they often trade mathematical interpretability and stability to achieve those metrics. 

To solve this, we designed a physics-constrained, first-order trainable nonlinear reaction-diffusion model. By replacing unconstrained convolution layers with geometric finite-differences, this model ensures strict mathematical well-posedness, maintaining coercivity and upholding the Maximum Principle.

## Key Features
*   **Parameter Efficiency**: Our architecture achieves a 97.6% reduction in parameter count compared to the TRDPD baseline. It requires only 128 parameters per stage, totaling just 1,024 parameters across the entire network.
*   **Rapid Training**: Because of the reduced complexity, training time is compressed from over 33 hours to approximately 20 minutes.
*   **Mathematical Stability**: The model strictly enforces positive diffusion coefficients by applying softplus activations to Radial Basis Function (RBF) networks. This completely prevents reverse-diffusion instabilities.
*   **Domain Generalizability**: The network avoids overfitting to dataset-specific textures by learning universal physical dynamics rather than relying on dense, unconstrained spatial filters.

## Architecture
The variational framework unrolls a continuous PDE into an 8-stage feed-forward network. Each stage consists of:
*   **Spatial Gradients**: Computed using four-directional (North, South, East, West) finite differences via tensor slicing instead of 2D convolutions.
*   **Diffusivity Calculation**: Edge-stopping and intensity-dependent coefficients are calculated using dedicated RBF networks.
*   **Proximal Projection**: An exact proximal mapping derived from the Csíszár I-divergence is applied to enforce Poisson data fidelity.

## Implementation Details
*   **Framework**: The feed-forward network was implemented from scratch using PyTorch.
*   **Training Strategy**: The model utilizes a greedy stage-wise training approach to maintain numerical stability. Each stage is trained for 50 epochs using the Adam optimizer with a learning rate of 0.001.
*   **Dataset**: Models are trained on 400 non-overlapping 180x180 patches extracted from the BSDS500 dataset. Evaluation is performed on a strictly held-out matched subset of 59 images from the BSD68 dataset.

## Results
This lightweight PDE model delivers highly competitive denoising performance at a fraction of the computational cost. Under low noise conditions (Peak = 40), it achieves a PSNR of 27.62 dB, successfully recovering image structure and preserving edge details. While first-order gradients naturally lead to some smoothing of fine textures in high-noise environments, the model successfully prevents the hallucination of false structural details.

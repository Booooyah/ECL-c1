# Empirical Cohomology Learning (ECL)

**Statistical Inference of the First Chern Class on Noisy Manifolds**

This repository contains the official Python implementation of the **Empirical Cohomology Learning (ECL)** framework, as introduced in our paper: *Empirical Cohomology Learning: Statistical Inference of the First Chern Class on Noisy Manifolds*.

## Overview

Inferring the macroscopic topological skeleton of empirical vector fields—specifically isolating causal singularities and bifurcations—constitutes a fundamental statistical challenge on unstructured manifolds. Classical continuous non-parametric spatial estimators inevitably suffer from **irreducible structural confounding** and **metric leakage**. Constrained by linear minimax lower bounds, noise-induced amplitude shrinkage structurally forces the topological annihilation of genuine singular features.

To bypass these fundamental limits, we introduce the **ECL framework**. By restricting inference to the discrete Čech cohomology of the first Chern class, ECL repurposes the algebraic connecting homomorphism from sheaf theory as a non-linear spatial truncation estimator. This repository demonstrates how ECL achieves non-asymptotic exact recovery within a deterministic discrete tolerance margin, and implements asymptotically valid spatial False Discovery Rate (FDR) control via the Chen-Stein Poisson limit to rigorously filter high-frequency topological artifacts.

## Core Theoretical Highlights

1. **Algebraic Truncation via Connecting Homomorphism**: Replaces continuous linear smoothing with a non-linear integer-valued discrete projection, strictly quotienting out bounded continuous metric leakage.
2. **Sub-resolution Annihilation**: Establishes a deterministic algebraic annihilation radius to cancel structurally coupled dipoles, satisfying strict asymptotic scale separation.
3. **Spatial Poisson Null Bath**: Derives an exact spatial null distribution using the Chen-Stein method. Surviving noise artifacts converge weakly to a Homogeneous Poisson Point Process (HPPP), enabling closed-form P-values and rigorous Benjamini-Hochberg FDR control.
4. **The "Zero-Speed Paradox"**: Analytically and empirically exposes the structural blindness of classical gradient-based Jacobian estimators on phase-normalized directional data.

## Repository Structure

* `models/`
  * `ECLC1.py` - Core ECL-c1 statistical topological inference engine
  * `baselines.py` - Classical continuous spatial statistics and Graph ML baselines
* `exp/`
  * `simexp1_compare.py` - Cross-paradigm blind benchmarking on closed manifolds
  * `simexp2_compare.py` - Evaluation of absolute boundary immunity on open manifolds
  * `simexp3.py` - Empirical null calibration, phase transition limits, and diagnostic circuit breaker
  * `simexp4_ablation.py` - Ablation study isolating the "Zero-Speed Paradox"
  * `realexp1.py` - Single-cell RNA velocity inference (Pancreas developmental trajectory)
  * `realexp2.py` - Lagrangian coherence tracking of mesoscale ocean turbulence
* `data/`
  * `ocean_5days.csv` - Placeholder for the AVISO satellite altimetry dataset (used in realexp2.py)

## Dependencies

The framework is implemented in pure Python. To reproduce the bioinformatics empirical experiment (`realexp1.py`), specific single-cell analysis libraries are required.

```bash
# Core mathematical and scientific computing libraries
pip install numpy scipy pandas matplotlib tqdm

# Bioinformatics dependencies (required ONLY for realexp1.py)
pip install scvelo scanpy
```

## Reproducing the Results

All scripts are configured to run out-of-the-box with standardized analytical configurations (e.g., nominal alpha=0.05), proving the parameter-free structural generalization of the ECL framework. All resulting publication-quality figures (in `.pdf` format) will be automatically saved in the execution directory.

### 1. Monte Carlo Simulations & Benchmarking
These scripts systematically generate synthetic manifolds, apply severe heteroscedastic noise (sigma=0.8), and execute comparative analyses against classical gradient-based spatial statistics and Markov random walk heuristics.

```bash
python exp/simexp1_compare.py
python exp/simexp2_compare.py
python exp/simexp4_ablation.py
```

### 2. Statistical Calibration and Phase Transitions
Validates Theorem 2 (Weak Convergence to the Spatial Poisson Null) and identifies the topological breakdown threshold using the raw BKT density diagnostic.

```bash
python exp/simexp3.py
```

### 3. Real-World Empirical Applications

**Single-cell Transcriptomics** (`realexp1.py`):
Automatically downloads the murine pancreatic endocrinogenesis dataset via `scvelo`. Statistically isolates developmental origins (+1 sources) and the critical lineage bifurcation (-1 saddle).

```bash
python exp/realexp1.py
```

**Geophysical Fluid Dynamics** (`realexp2.py`):
Performs independent daily spatial inferences on mesoscale ocean turbulence to validate emergent Lagrangian spatiotemporal coherence. *(Note: Requires the AVISO geostrophic velocity CSV dataset placed in `data/ocean_5days.csv`)*

```bash
python exp/realexp2.py
```

## Citation

If you find this framework or the theoretical insights useful for your research, please consider citing our paper:

```bibtex
@article{ecl_2026,
  title={Empirical Cohomology Learning: Statistical Inference of the First Chern Class on Noisy Manifolds},
  author={[Anonymized for Double-Blind Review]},
  journal={Submitted to the Annals of Statistics},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

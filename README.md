# FGP_2023

### N.B. in progress.

These notebooks reproduce the implementation and analysis of a biophysical model for ATAC-seq data analysis. 

The notebooks are arranged as follows:

- moments.ipynb: Calculate transcriptome and chromatin-state moments and compare to simluated results for a two-gene system.
- distinguishability_simulation.ipynb: Fit our biophysical model to simulated results and compare parameter identifiability for different modalities.
- noise.ipynb: Illustration of technical noise implementation.
- BIC_analysis: Fit models with peak-peak correlations to ATAC data and compare Bayesian Information Criteria with a simple model without correlations.

The following scripts contain helper functions arranged as follows:

- construct_H.py: Create transition matrix for Ising chromatin model.
- simulation.py: Simulate RNA counts and chromatin configurations from model.
- noise.py: Technical noise implementation.

We also include a script to show how the six-site loci for ATAC-seq analysis were selected:

- loci_selection.py

The observed chromatin accessibility at these loci from the processed ATAC-seq data are included in sub-folders.

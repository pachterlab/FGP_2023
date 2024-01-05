# FGP_2023

These notebooks reproduce the implementation and analysis of a biophysical model for ATAC-seq data analysis. 

The notebooks are arranged as follows:

- moments.ipynb: Calculate transcriptome and chromatin-state moments and compare to simluated results for a two-gene system.
- distinguishability_simulation.ipynb: Fit our biophysical model to simulated results and compare parameter identifiability for different modalities.
- noise.ipynb: Illustration of technical noise implementation.
- loci_statistics.ipynb: Perform exploratory data analysis of ATAC-seq data. Plot site-site correlations and means.
- BIC_analysis.ipynb: Fit models with peak-peak correlations to ATAC data and compare Bayesian Information Criteria with a simple model without correlations.
- toy_systems.ipynb: Illustrate toy systems which have transcript correlations greater than chromatin site correlations.

The following scripts contain helper functions arranged as follows:

- construct_H.py: Create transition matrix for Ising chromatin model.
- simulation.py: Simulate RNA counts and chromatin configurations from model.
- noise.py: Technical noise implementation.
- moments.py: Functions for calculating moments.

We also include a script to show how the six-site loci for ATAC-seq analysis were selected, using one of the datasets as an example.

- loci_selection.py

The observed chromatin accessibility at these loci from the processed ATAC-seq data are included in sub-folders.

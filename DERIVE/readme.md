# DERIVE
This is the official implementation of the paper: Disentangled evolutionary representations enable cross-virus antigenic forecasting


# Overview
DERIVE learns a disentangled latent representation of viral evolutionary landscapes to anticipate antigenic evolution and immune escape. Using only pre-pandemic sequence data, it integrates sequence homology, physicochemical descriptors and structural context to separate antigenicity from other evolutionary pressures, reconstruct evolutionary trajectories, forecast immune escape and prevalence trends, and generate interpretable mutation-level effect maps that generalize across diverse viruses.


# Requirements
This repository uses only Python. Its package dependencies are listed below:
- python=3.8
- pytorch==2.4.0
- torch-geometric=2.6.1
- cudatoolkit=11.0
- pandas=1.2.4
- scikit-learn=0.24.1
- numpy=1.22.4
- matplotlib=3.7.5
- seaborn=0.13.2
- tqdm

# MSA creation
For a single database, the detailed procedure for building the multiple sequence alignment is as follows:

Step 1:
Run jackhmmer (5 iterations, with the initial bit-score threshold set to 0.3×L, where L is the length of the protein sequence) to obtain the initial hits.

Step 2:
From the hits, extract sequences with bit score ≥ b×L (where b = b₀ and the initial threshold b₀ is set to 0.5). Compute the coverage Lcov of the extracted sequences (the fraction of positions in the target sequence that are covered; a position is considered covered if at least one sequence has an amino acid at that position) and the effective number of sequences Neff (effective number of sequences) based on the Hamming distance.

Step 3: Check the following conditions:

- Condition A: Lcov ≥ 0.9×L
- Condition B: Neff ≥ 2×L

If only Condition A is not satisfied, increase the bit-score threshold from b×L to (b + 0.01)×L and repeat Steps 2–3.

If only Condition B is not satisfied (N < 10L), decrease the bit-score threshold from b×L to (b − 0.01)×L and repeat Steps 2–3.

If both Condition A and Condition B are not satisfied, move to the relaxed-criteria strategy:

- Relaxed Condition A: Lcov ≥ 0.8×L 
- Relaxed Condition B: Neff ≥ L

In general, it is uncommon that both relaxed conditions cannot be met; if this occurs, the criteria are further relaxed.


# Data requirements
Training DERIVE models from scratch and computing DERIVE scores requires only the multiple sequence alignments (MSAs) of the corresponding proteins.
For fair comparison with other methods and to facilitate reproduction of our results, we also provide a set of precomputed MSAs for the proteins used in our experiments, available in the `./data/MSA/` directory. 

# Run Our Model 
Simply run the following command to train
```
python main_train.py --GCN_pretrained_initialization {...} --d0_similarity_threshold {...} --eta_balancing_strength{...} ...
```

# Model Evaluation
```
python test.py --checkpoint_path ./checkpoints/latest_model.pth --result_folder ./results
```


# Reference 
If you use this code, please cite the following paper:
```

```
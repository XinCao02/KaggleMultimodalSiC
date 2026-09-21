# SS-OPM comparator

The comparator adapts the architecture and target objective of
[Shuji Suzuki’s first-place solution](https://github.com/shu65/open-problems-multimodal/tree/2a735ad9ab6ad8c391769f1f0cac11301539aa91).
It is not a reproduction of the complete winning ensemble. Attribution and the
upstream MIT license are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

| Component | This implementation |
|---|---|
| RNA inputs | Same training-only PCA, selected genes and QC features as ResMLP |
| Architecture | Width 2,048; encoder, five decoder blocks and six latent prediction heads |
| Targets | Within-cell percentile clipping, median scaling, standardization, donor-day median subtraction and 128-component SVD, fitted on training cells |
| Objective | Mean head correlation loss plus latent L1; 10-epoch warm-up and quadratic decay |
| Optimization | Original tuned Adam hyperparameters and OneCycle; batch 256 instead of 64 |
| Evaluation | Separate training, validation and test partitions; one comparator seed |

Original imputation, pathway features, metadata inputs, batch-specific fine-tuning
and the winning ensemble are omitted. Other differences include NumPy SVD,
a near-zero median guard, global optimizer weight decay instead of the original
parameter-group exclusions, and keeping the last partial minibatch.

The published competition ranking combines CITE-seq and Multiome predictions;
our local CITE-seq correlations are not official leaderboard scores.

# Notebooks

Use this folder for notebook-driven project work.

```text
01_cp_surrogate_model.ipynb
02_su2_vs_kaggle_comparison.ipynb
03_neural_network_comparison.ipynb
05_aoa_mach_neural_network_comparison.ipynb
```

The notebook story is:

- `01` Kaggle baseline Cp surrogate models
- `02` dense SU2 dataset validation against Kaggle
- `03` neural-network comparison against the stronger classical baseline
- `05` neural-network comparison on the AoA x Mach SU2 extension dataset

Keep notebooks short and story-focused. Move reusable loading, feature
engineering, evaluation, and plotting code into `cfd_ml/`.

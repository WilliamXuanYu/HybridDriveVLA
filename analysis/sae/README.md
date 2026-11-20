# Representation and SAE analyses

The training and plotting scripts accept feature, checkpoint and output paths as command-line arguments.

Download the archived SAE weights from [Hugging Face](https://huggingface.co/XuanYuer/HybridDriveVLA) by running the following at the repository root:

```bash
hf download XuanYuer/HybridDriveVLA --include 'analysis/sae/**/*.pt' --local-dir .
```

| Archived results | Recovered entry point |
| --- | --- |
| [ViT-L sweeps](VIT_large/) | [analyze_sae_shared_unique.py](scripts/analyze_sae_shared_unique.py) trains shared/unique autoencoders, runs loss-weight sweeps and re-evaluates saved checkpoints. |
| [Sweep figures and random controls](VIT_large/sae_experiments/) | [visualize_sae_results.py](scripts/visualize_sae_results.py) renders the paired and shuffled result CSVs. |
| [Multi-model CKA, CCA and pairwise SAE](multi_model_analysis/) | [analyze_multi_models.py](scripts/analyze_multi_models.py) runs the comparisons and generates the report. |
| [Multi-model representation plots](multi_model_analysis/) | [visualize_multi_model_space.py](scripts/visualize_multi_model_space.py), [visualize_multi_model_advanced.py](scripts/visualize_multi_model_advanced.py), [visualize_multi_model_v2.py](scripts/visualize_multi_model_v2.py) and [plot_cka_comparison.py](scripts/plot_cka_comparison.py) use token-aligned features. |
| [CCA spectra](VIT_large/) | [analyze_pca_whitening_cca.py](scripts/analyze_pca_whitening_cca.py) and [analyze_backbone_cka_cca.py](scripts/analyze_backbone_cka_cca.py) run whitened and direct CCA, respectively. |
| [Train-split SAE variants](VIT_large_train/) | Historical weights and figures. Their exact input caches and run settings need to be matched separately. |

## Inputs and setup

Run the examples below from `analysis/sae/`. Install the [analysis dependencies](requirements.txt) in the environment used for ReCogDrive, or in a separate PyTorch environment. The historical checkpoints store fitted scikit-learn `StandardScaler` objects alongside tensors. Use trusted local checkpoints, feature pickles and NPZ caches.

From the repository root, run the PDM evaluator with `CHECKPOINT` set and `save_features=true feature_save_path=/path/to/export`. It writes `recogdrive_mapped_features/<token>/mapped_features.pkl` and `recogdrive_dit_output/<token>/dit_output.pkl`. The scripts align features by NAVSIM token and average non-channel dimensions per scene. Use the same split for each model. In the historical experiment names, `backbone` refers to mapped features at the planner input.

Pairwise entry points accept either `--x-dir` and `--y-dir` or an existing `--cache` with arrays `X`, `Y`, and `common_ids`. The feature exports and aligned caches are separate inputs. The ReCogDrive VLM remains a historical comparison reference and requires its corresponding feature exports for VLM comparisons.

## Re-run SAE and plot the archived results

```bash
# Train one SAE between two visual policies.
python scripts/analyze_sae_shared_unique.py \
  --x-dir /path/to/resnet101/recogdrive_mapped_features \
  --y-dir /path/to/vit_large/recogdrive_mapped_features \
  --output ../../outputs/sae/resnet101_vit

# Re-evaluate the original VLM vs ViT-L mapped-feature checkpoint with its original cache.
python scripts/analyze_sae_shared_unique.py --eval \
  --cache /path/to/aligned_features_cache_backbone.npz \
  --model VIT_large/sae_experiments/sae_model_backbone_crossRecon0.5.pt \
  --output ../../outputs/sae/recheck

# Rebuild the sweep figures from the archived CSVs.
python scripts/visualize_sae_results.py \
  --csv VIT_large/sae_experiments/batch_experiment_results_updated.csv \
  --random-csv VIT_large/sae_experiments_random/batch_experiment_results_random.csv \
  --output ../../outputs/sae/sweep_figures
```

Use `--batch` for the sweep over normalized versus raw invariance MSE and five cross-reconstruction weights. Select `--feature-type dit` with DiT inputs. For the archived ViT-L sweeps, the original experiment workspace contains the inputs below.

| Experiment | Original path relative to the experiment workspace |
| --- | --- |
| Mapped features | `Feature/VIT_large/aligned_features_cache_backbone.npz` |
| DiT features | `Feature/VIT_large/aligned_features_cache_vit_large_dit.npz` |
| Four-model visualization | `Feature/multi_model_analysis/aligned_cache/` |

Use `--eval --model-dir DIRECTORY` to re-evaluate a full sweep. For the historical shuffled control, use `--random --random-seed 42`. Add `--random-prefix-size 12146` for DiT to reproduce the original permutation order, which shuffled the 12,146 mapped-feature scenes first. The plotter accepts multiple per-layer CSV paths after either CSV option.

## Compare multiple models

```bash
python scripts/analyze_multi_models.py \
  --model ViT-Large /path/to/vit_large/recogdrive_mapped_features \
  --model ResNet101 /path/to/resnet101/recogdrive_mapped_features \
  --model EVA-CLIP /path/to/eva_clip/recogdrive_mapped_features \
  --output ../../outputs/sae/multi_model

python scripts/plot_cka_comparison.py \
  --cache-dir /path/to/aligned_cache \
  --output ../../outputs/sae/cka
```

Add `--model VLM /path/to/vlm/recogdrive_mapped_features` first to match the archived four-model ordering. The four-model plotters read the aligned caches generated by `visualize_multi_model_space.py`, which accepts one `--model NAME EXPORT_ROOT` per model and can use an existing `--cache-dir`. The advanced plotter accepts `--model-dir` to add the VLM/ViT SAE latent figure. It uses UMAP when `umap-learn` is installed and PCA otherwise.

## Verification

Using the original 12,146-scene caches, the paired mapped-feature, paired DiT, and shuffled DiT checkpoints at cross-reconstruction weight 0.5 reproduced the archived original CKA, shared CKA, and X/Y reconstruction R² within `1e-6`. The plotter regenerated all seven paired and shuffled sweep figures from the archived CSVs. A small run also exercised single-model training, the ten-configuration sweep and re-evaluation, and the multi-model report and plotting entry points.

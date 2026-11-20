# HybridDriveVLA

[From Representational Complementarity to Dual Systems: Synergizing VLM and Vision-Only Backbones for End-to-End Driving](https://arxiv.org/abs/2602.10719) 

This initial release contains the vision-only backbone experiments and representation analyses from the paper. The encoders use the front camera at 224×224 and share a NAVSIM pixel cache. The code supports imitation learning (IL), Diffusion-GRPO reinforcement learning (RL), PDM evaluation and intermediate feature export. We are organizing the remaining materials and will update the repository incrementally.

For the original VLM-based planner, its pretrained checkpoints and reproduction instructions, see [ReCogDrive](https://github.com/xiaomi-research/recogdrive). This release builds on ReCogDrive and [NAVSIM](https://github.com/autonomousvision/navsim). See [LICENSE](LICENSE) for the Apache 2.0 code license.

## Models and weights

Download the checkpoints from [Hugging Face](https://huggingface.co/XuanYuer/HybridDriveVLA). The policy table covers backbones with both IL and RL checkpoints.

| Encoder | `BACKBONE` | Default pretrained backbone | IL checkpoints | RL checkpoints |
| --- | --- | --- | --- | --- |
| ResNet-50 | `resnet50` | [TorchVision ImageNet-1K V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) | `resnet_50_8237.ckpt` | `resnet50_grpo_8602.ckpt` |
| ResNet-101 | `resnet101` | [TorchVision ImageNet-1K V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet101.html) | `resnet_101_8463.ckpt` | `resnet101_grpo_8769.ckpt` |
| EVA-CLIP Base/16 | `eva_clip_base` | [timm EVA02 CLIP](https://huggingface.co/timm/eva02_base_patch16_clip_224.merged2b) | `eva_clip_base_8466.ckpt` | `eva_clip_base_grpo.ckpt` |
| ViT-B/16 | `vit_b_16` | [TorchVision ImageNet-1K V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) | `vit_b16_joint_il.ckpt`, `vit_b16_diffusion_finetuned_il.ckpt` | `vit_b16_grpo_8821.ckpt` |
| ViT-L/16 | `vit_large` | [timm ViT-L/16 AugReg](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k) | `vit_l_8856.ckpt` | `vit_l_grpo_1.ckpt`, `vit_l_grpo_8883.ckpt` |

Run these commands from the repository root to download just the ResNet-50 checkpoints used below:

```bash
pip install -U huggingface_hub
hf download XuanYuer/HybridDriveVLA \
  checkpoints/resnet_50_8237.ckpt checkpoints/resnet50_grpo_8602.ckpt \
  --local-dir .
```

To download all policy weights, use `hf download XuanYuer/HybridDriveVLA --include 'checkpoints/*.ckpt' --local-dir .`. The Hub also hosts the archived SAE weights. See the [SAE instructions](analysis/sae/README.md) for the download command and required inputs.

## Setup

Use a Linux environment with CUDA and Python 3.9–3.11. Install PyTorch and torchvision suitable for your CUDA installation, then install the project. Download the NAVSIM v1 dataset and nuPlan maps following the [NAVSIM documentation](https://github.com/autonomousvision/navsim).

```bash
pip install -e .
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NAVSIM_EXP_ROOT="$PWD/outputs"
# Optional. Defaults to one process on one visible GPU.
export GPUS=1
```

The dataset root should contain `navsim_logs/{trainval,test}` and `sensor_blobs/{trainval,test}`. The launcher takes Hydra overrides after the operation and backbone. Run the examples from the repository root.

## Train and evaluate

```bash
# Generate shared training pixels and the DiffGRPO reward cache.
scripts/run.sh cache-pixels resnet50
SPLIT=navtrain scripts/run.sh cache-metrics resnet50

# Jointly train the backbone and diffusion planner.
scripts/run.sh il resnet50

# Train DiffGRPO from IL. REFERENCE_CHECKPOINT can select another IL checkpoint.
export CHECKPOINT="$PWD/checkpoints/resnet_50_8237.ckpt"
scripts/run.sh rl resnet50

# Build the navtest metric cache and score an RL model.
SPLIT=navtest scripts/run.sh cache-metrics resnet50
export CHECKPOINT="$PWD/checkpoints/resnet50_grpo_8602.ckpt"
scripts/run.sh eval resnet50

# Export representations aligned by NAVSIM scene token.
scripts/run.sh eval resnet50 save_features=true feature_save_path="$PWD/outputs/resnet50_rl_features"
```

`PIXEL_CACHE` and `METRIC_CACHE` accept existing cache roots. `BACKBONE_WEIGHTS` allows IL to start from a local vision initialization. Otherwise torchvision or timm fetches its pretrained weights. Full policy checkpoints provide their own backbone weights when evaluated. The launcher gives general training defaults. Override batch size, epoch count, devices and optimizer settings as needed for a specific run.

Use `best_of_n=3` to evaluate the simulator-selected best of three candidate trajectories. Use `save_predictions=true` to export the predictions, and single-sample mode to export intermediate features.

## Analysis

- [Archived NAVSIM ablation results](analysis/ablation/README.md)
- [SAE training, re-evaluation, weights and figures](analysis/sae/README.md)

The VLM appears as a historical comparator in the archived results and representation analyses. The trainable policies in this repository use the visual encoders listed above.

## Citation

For this release, please cite [the HybridDriveVLA paper](https://arxiv.org/abs/2602.10719). For work based on the original planner, please also cite ReCogDrive.

```bibtex
@article{li2025recogdrive,
  title={ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving},
  author={Li, Yongkang and Xiong, Kaixin and Guo, Xiangyu and Li, Fang and Yan, Sixu and Xu, Gangwei and Zhou, Lijun and Chen, Long and Sun, Haiyang and Wang, Bing and others},
  journal={arXiv preprint arXiv:2506.08052},
  year={2025}
}
```

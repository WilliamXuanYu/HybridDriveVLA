# Modified for the ReCogDrive visual backbone ablation release.
from typing import Tuple
from pathlib import Path
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"




def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list]:
    # Handle both Dataset (2 values) and CacheOnlyDataset (3 values) returns
    if len(batch[0]) == 3:
        features_list, targets_list, tokens_list = zip(*batch)
    else:
        features_list, targets_list = zip(*batch)
        tokens_list = []  # Initialize empty tokens list for compatibility

    # Optimize: reduce unnecessary tensor operations and data transfer
    # Keep tensors on their original device (CPU for most, GPU if already there)
    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0)
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0)
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0)

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0)

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature
    }

    # Handle both cached hidden states and raw pixel values
    if 'last_hidden_state' in features_list[0]:
        # Use efficient padding for variable-length hidden states
        last_hidden_state = rnn_utils.pad_sequence(
            [features['last_hidden_state'] for features in features_list],
            batch_first=True,
            padding_value=0.0
        )
        features['last_hidden_state'] = last_hidden_state
    elif 'pixel_values' in features_list[0]:
        # Optimized pixel value stacking for ResNet joint training
        # Each sample: [1, 3, 224, 224], stack to [B, 3, 224, 224]
        pixel_values = torch.cat([features['pixel_values'] for features in features_list], dim=0)
        features['pixel_values'] = pixel_values

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    # For joint training (train_backbone=True), disable caching since features change each epoch
    effective_cache_path = None if getattr(cfg, 'disable_caching', False) else cfg.cache_path
    
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=effective_cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=effective_cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()  # Load checkpoint weights (including backbone)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    
    # Optimized validation dataloader with reduced workers for faster validation
    val_dataloader_params = dict(cfg.dataloader.params)
    val_dataloader_params['num_workers'] = max(4, val_dataloader_params.get('num_workers', 8) // 4)
    val_dataloader_params['prefetch_factor'] = 2
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **val_dataloader_params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    # Adjust ModelCheckpoint to match validation frequency
    checkpoint_every_n = cfg.trainer.params.get('check_val_every_n_epoch', 1)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        mode='min', 
        save_top_k=5,
        every_n_epochs=checkpoint_every_n
    )])

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()

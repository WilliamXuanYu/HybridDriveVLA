# Modified for the ReCogDrive visual backbone ablation release.
from typing import Any, Dict, List, Union
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import torch
import torch.distributed as dist
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    # Get best_of_n parameter (default to 1 for standard evaluation)
    best_of_n = getattr(cfg, 'best_of_n', 1)
    if dist.get_rank() == 0 and best_of_n > 1:
        logger.info(f"Best-of-{best_of_n} evaluation enabled")

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    
    # Check if feature saving is enabled
    save_features = cfg.save_features
    if save_features and best_of_n != 1:
        raise ValueError('Feature export requires best_of_n=1.')
    feature_save_path = None
    mapped_features_path = None
    dit_output_path = None
    if save_features:
        feature_save_path = Path(getattr(cfg, 'feature_save_path', '/tmp/features'))
        # Create separate directories for mapped_features and dit_output
        mapped_features_path = feature_save_path / 'recogdrive_mapped_features'
        dit_output_path = feature_save_path / 'recogdrive_dit_output'
        mapped_features_path.mkdir(parents=True, exist_ok=True)
        dit_output_path.mkdir(parents=True, exist_ok=True)
        if dist.get_rank() == 0:
            logger.info(f"Feature saving enabled. Saving to: {feature_save_path}")
            logger.info(f"  - Mapped features: {mapped_features_path}")
            logger.info(f"  - DiT output: {dit_output_path}")
    
    # Check if prediction saving is enabled
    save_predictions = getattr(cfg, 'save_predictions', False)
    predictions_save_path = None
    if save_predictions:
        predictions_save_path = Path(getattr(cfg, 'predictions_save_path', '/tmp/predictions'))
        predictions_save_path = predictions_save_path / 'recogdrive_predictions'
        predictions_save_path.mkdir(parents=True, exist_ok=True)
        if dist.get_rank() == 0:
            logger.info(f"Prediction saving enabled. Saving to: {predictions_save_path}")
            logger.info(f"  - Output shape: [8, 3] for single sample, [N, 8, 3] for best-of-N")

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )


    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate) 
    
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        if dist.get_rank() == 0:
            logger.info(f"Rank {dist.get_rank()} processing scenario {idx+1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}")

        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            # Best-of-N evaluation
            if best_of_n > 1:
                # Generate N trajectory samples
                trajectories = agent.compute_trajectory(agent_input, token=token, num_samples=best_of_n)
                
                # Save predictions if enabled (shape: [N, 8, 3])
                if save_predictions:
                    import numpy as np
                    token_pred_dir = predictions_save_path / token
                    token_pred_dir.mkdir(parents=True, exist_ok=True)
                    pred_file = token_pred_dir / "predictions.pkl"
                    # Stack all trajectory poses into [N, 8, 3] array
                    all_poses = np.stack([traj.poses for traj in trajectories], axis=0)
                    with open(pred_file, 'wb') as f:
                        pickle.dump(all_poses, f)
                
                # Compute PDM score for each trajectory
                all_scores = []
                all_pdm_results = []
                for i, traj in enumerate(trajectories):
                    result = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=traj,
                        future_sampling=simulator.proposal_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                    all_scores.append(result.score)
                    all_pdm_results.append(result)
                    # Store individual trajectory scores
                    score_row[f"score_sample_{i+1}"] = result.score
                
                # Find the best trajectory
                best_idx = all_scores.index(max(all_scores))
                best_pdm_result = all_pdm_results[best_idx]
                
                # Update with best trajectory's full results (as the main score)
                score_row.update(asdict(best_pdm_result))
                score_row["best_sample_idx"] = best_idx + 1
                score_row["mean_score"] = sum(all_scores) / len(all_scores)
                score_row["min_score"] = min(all_scores)
                score_row["max_score"] = max(all_scores)
            else:
                # Standard single-sample evaluation
                if save_features:
                    result = agent.compute_trajectory(agent_input, return_intermediate_features=True, token=token)
                    if isinstance(result, tuple):
                        trajectory, predictions = result
                        # Save features following the original structure:
                        # recogdrive_mapped_features/{token}/mapped_features.pkl
                        # recogdrive_dit_output/{token}/dit_output.pkl
                        
                        # Save mapped features
                        if "mapped_features" in predictions:
                            token_mapped_dir = mapped_features_path / token
                            token_mapped_dir.mkdir(parents=True, exist_ok=True)
                            mapped_feat_path = token_mapped_dir / "mapped_features.pkl"
                            with open(mapped_feat_path, 'wb') as f:
                                pickle.dump(predictions["mapped_features"].cpu(), f)
                        
                        # Save DiT output
                        if "dit_output" in predictions:
                            token_dit_dir = dit_output_path / token
                            token_dit_dir.mkdir(parents=True, exist_ok=True)
                            dit_output_file = token_dit_dir / "dit_output.pkl"
                            with open(dit_output_file, 'wb') as f:
                                pickle.dump(predictions["dit_output"].cpu(), f)
                    else:
                        trajectory = result
                else:
                    trajectory = agent.compute_trajectory(agent_input, token=token)
                
                # Save single-sample prediction if enabled (shape: [8, 3])
                if save_predictions:
                    token_pred_dir = predictions_save_path / token
                    token_pred_dir.mkdir(parents=True, exist_ok=True)
                    pred_file = token_pred_dir / "predictions.pkl"
                    with open(pred_file, 'wb') as f:
                        pickle.dump(trajectory.poses, f)
                
                # Compute PDM score for single sample
                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
                score_row.update(asdict(pdm_result))
        
            score_row['rank'] = dist.get_rank()
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='gloo',
        world_size=world_size,
        rank=rank,
    )
    
    torch.cuda.set_device(local_rank)

    build_logger(cfg)


    scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_config=SensorConfig.build_no_sensors(),
        )
    if rank == 0:
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        tokens_to_evaluate = sorted(tokens_to_evaluate)  
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    else:
        tokens_to_evaluate = []

    token_payload = [tokens_to_evaluate]
    dist.broadcast_object_list(token_payload, src=0)
    tokens_to_evaluate = token_payload[0]


    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    if not tokens_to_evaluate:
        raise ValueError("The selected scenes and metric cache have no common tokens.")
    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token] 
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],
        })

    local_results = run_pdm_score(data_points) if data_points else []
    gathered_results = [None] * world_size if rank == 0 else None
    dist.gather_object(local_results, gathered_results, dst=0)
    dist.destroy_process_group()

    if rank == 0:
        final_results = [row for shard in gathered_results for row in shard]
        pdm_score_df = pd.DataFrame(final_results)
        if not pdm_score_df["valid"].any():
            raise RuntimeError("Every PDM scenario failed. See the scenario errors above.")

        num_sucessful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
        
        # Identify columns to exclude from averaging
        exclude_cols = ["token", "valid", "rank", "best_sample_idx"]
        cols_to_average = [c for c in pdm_score_df.columns if c not in exclude_cols]
        
        average_row = pdm_score_df[cols_to_average].mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = pdm_score_df["valid"].all()
        average_row["rank"] = "0"
        if "best_sample_idx" in pdm_score_df.columns:
            average_row["best_sample_idx"] = ""
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        
        # Get best_of_n for filename
        best_of_n = getattr(cfg, 'best_of_n', 1)
        if best_of_n > 1:
            csv_filename = f"{timestamp}_best_of_{best_of_n}.csv"
        else:
            csv_filename = f"{timestamp}.csv"
        
        pdm_score_df.to_csv(save_path / csv_filename)

        # Log results
        log_msg = f"""
            Finished running evaluation.
                Number of successful scenarios: {num_sucessful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
                Final average score (best-of-{best_of_n}): {pdm_score_df['score'].mean():.4f}.
                Results are stored in: {save_path / csv_filename}."""
        
        if best_of_n > 1 and "mean_score" in pdm_score_df.columns:
            mean_of_means = pdm_score_df['mean_score'].iloc[:-1].mean()  # Exclude average row
            log_msg += f"""
                Average of mean scores: {mean_of_means:.4f}.
                Improvement from best-of-{best_of_n}: {pdm_score_df['score'].mean() - mean_of_means:.4f}."""
        
        logger.info(log_msg)


if __name__ == "__main__":
    main()

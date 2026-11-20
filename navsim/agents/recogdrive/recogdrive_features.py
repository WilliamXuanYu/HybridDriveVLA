# Modified for the ReCogDrive visual backbone ablation release.
from typing import Dict, Optional
import torch
import numpy as np
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .recogdrive_backbone import RecogDriveBackbone

class ReCogDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False, ):
        """Build the shared 224px pixel cache or compute frozen visual features."""
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.cache_mode = cache_mode
        self.model_type = model_type.lower() if model_type else None
        self.device = device
        self.backbone = None

        if self.cache_hidden_state:
            if not self.model_type:
                raise ValueError("When cache_hidden_state=True, `model_type` must be provided.")
            self.backbone = RecogDriveBackbone(
                model_type=model_type,
                checkpoint_path=checkpoint_path if checkpoint_path is not None else "",
                device=device
            )

    def get_unique_name(self) -> str:
        return f"{self.model_type}_hidden_feature" if self.cache_hidden_state else "resnet34_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        high_command_one_hot = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        status_feature = torch.cat([
            high_command_one_hot.clone(),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)


        pixels = self._load_resnet_image_fast(cameras[-1].cam_f0.image)
        features = {
            "history_trajectory": history_trajectory,
            "high_command_one_hot": high_command_one_hot,
            "status_feature": status_feature,
        }
        if self.cache_hidden_state:
            output = self.backbone(pixels.to(self.device)).hidden_states[-1]
            features["last_hidden_state"] = output.squeeze(0).float().cpu()
        else:
            features["pixel_values"] = pixels
        return features

    def _load_resnet_image_fast(self, image_input):
        """
        Fast image loading without unnecessary dynamic preprocessing.
        For ResNet and ViT, we only need a single 224x224 patch.
        """
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        # Convert to PIL image
        if isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_input).convert('RGB')
        elif torch.is_tensor(image_input):
            arr = image_input.detach().cpu().numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr).convert('RGB')
        else:
            # Assume it's a file path
            image = Image.open(image_input).convert('RGB')
        
        # Simple resize and normalize (avoid dynamic_preprocess overhead)
        transform = T.Compose([
            T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        pixel_values = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        return pixel_values


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}

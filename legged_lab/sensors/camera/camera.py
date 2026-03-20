# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.log
import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import Camera as BasedCamera
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from .camera_cfg import CameraCfg


class Camera(BasedCamera):
    def __init__(self, cfg: "CameraCfg"):
        self._is_headless = carb.settings.get_settings().get("/app/runLoops/main/headless")
        if self._is_headless:
            cfg.debug_vis = False

        super().__init__(cfg)
        self.cfg = cfg
        self.sensor_noise = self.cfg.sensor_noise

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        # Apply noise to the depth images after updating the buffers
        self._apply_noise()
        self._apply_range_limits()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "point_cloud_visualizer"):
                if self.cfg.visualizer_cfg is None:
                    omni.log.warn(
                        f"Missing 'visualizer_cfg' in configuration for Camera '{self.cfg.prim_path}'. Unable to create"
                        " point cloud visualization."
                    )
                    return
                self.point_cloud_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

            if hasattr(self, "point_cloud_visualizer"):
                self.point_cloud_visualizer.set_visibility(True)
        else:
            if hasattr(self, "point_cloud_visualizer"):
                self.point_cloud_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        # print("Entering debug visualization callback for camera.")
        if self.frame[0] < 5:
            return
        if not hasattr(self, "point_cloud_visualizer") or not self.point_cloud_visualizer.is_visible():
            return

        if "distance_to_image_plane" not in self.data.output:
            omni.log.warn_once(
                f"Debug visualization of camera '{self.cfg.prim_path}' requires 'distance_to_image_plane' data type."
            )
            return

        depth_images = self._data.output["distance_to_image_plane"]
        intrinsic_matrices = self._data.intrinsic_matrices
        pos_w, quat_w_opengl = self._view.get_world_poses()

        num_envs, height, width, _ = depth_images.shape
        decimation = getattr(self.cfg.visualizer_cfg, "decimation", 10)

        v, u = torch.meshgrid(
            torch.arange(0, height, decimation, device=self.device),
            torch.arange(0, width, decimation, device=self.device),
            indexing="ij",
        )

        depth_decimated = depth_images[:, v, u].squeeze(-1)
        valid_mask = torch.isfinite(depth_decimated)
        if not torch.any(valid_mask):
            self.point_cloud_visualizer.visualize(translations=torch.empty((0, 3), device=self.device))
            return

        f_x = intrinsic_matrices[:, 0, 0].view(num_envs, 1, 1)
        f_y = intrinsic_matrices[:, 1, 1].view(num_envs, 1, 1)
        c_x = intrinsic_matrices[:, 0, 2].view(num_envs, 1, 1)
        c_y = intrinsic_matrices[:, 1, 2].view(num_envs, 1, 1)

        x_cam = (u - c_x) * depth_decimated / f_x
        y_cam = -((v - c_y) * depth_decimated / f_y)
        z_cam = -depth_decimated

        points_cam_opengl = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        points_flat = points_cam_opengl.view(num_envs, -1, 3)

        num_points = points_flat.shape[1]
        quat_exp = quat_w_opengl.unsqueeze(1).expand(-1, num_points, -1)  # [num_envs, N, 4]
        pos_exp = pos_w.unsqueeze(1).expand(-1, num_points, -1)  # [num_envs, N, 3]

        points_world = quat_apply(quat_exp, points_flat) + pos_exp
        final_points = points_world.view(-1, 3)[valid_mask.view(-1)]
        self.point_cloud_visualizer.visualize(translations=final_points)

    def _apply_noise(self):
        # print("Entering apply noise for camera.")
        if not self.sensor_noise.enable:
            return

        if "distance_to_image_plane" not in self._data.output:
            omni.log.warn_once(
                f"Camera '{self.cfg.prim_path}' does not have 'distance_to_image_plane' data type. Noise will not be"
                " applied."
            )
            return

        depth_iamges = self._data.output["distance_to_image_plane"]
        if self.sensor_noise.mode in ["gaussian", "combined"]:
            depth_iamges = self._apply_gaussian_noise(depth_iamges)
        if self.sensor_noise.mode in ["dropout", "combined"]:
            depth_iamges = self._apply_dropout_noise(depth_iamges)

        self._data.output["distance_to_image_plane"] = depth_iamges

    def _apply_gaussian_noise(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian noise to the depth images."""

        std_dev = self.sensor_noise.depth_std
        std_dev += depth_images * self.sensor_noise.depth_std_multiplier

        noise = torch.randn_like(depth_images) * std_dev

        noisy_depth = depth_images + noise
        return noisy_depth

    def _apply_dropout_noise(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Applies dropout noise to the depth data."""
        dropout_mask = torch.rand_like(depth_images) < self.sensor_noise.dropout_prob

        noisy_depth = depth_images.clone()
        noisy_depth[dropout_mask] = self.sensor_noise.dropout_value

        return noisy_depth

    def _apply_range_limits(self):
        valid_range_mask = (self._data.output["distance_to_image_plane"] >= self.cfg.min_range) & (
            self._data.output["distance_to_image_plane"] <= self.cfg.max_range
        )

        self._data.output["distance_to_image_plane"] = torch.where(
            valid_range_mask, self._data.output["distance_to_image_plane"], torch.inf
        )

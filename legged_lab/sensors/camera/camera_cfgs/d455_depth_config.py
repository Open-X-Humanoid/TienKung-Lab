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

import math

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from legged_lab.sensors.camera import CameraCfg, SensorNoiseCfg, TiledCameraCfg

D455_DEPTH_HFOV_DEG = 87.0
D455_HORIZONTAL_APERTURE_CM = 2.4
D455_FOCAL_LENGTH_CM = (D455_HORIZONTAL_APERTURE_CM / 2.0) / math.tan(math.radians(D455_DEPTH_HFOV_DEG) / 2.0)


@configclass
class D455Camera:
    enable_depth_camera = True
    debug_vis = False

    width: int = 480
    height: int = 270
    max_range: float = 15.0
    min_range: float = 0.2

    data_types: list[str] = ["distance_to_image_plane"]

    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
        pos=(0.10, 0.0, 0.03), rot=(0.707, 0.0, 0.707, 0.0), convention="ros"
    )

    spawn: PinholeCameraCfg = PinholeCameraCfg(
        focal_length=D455_FOCAL_LENGTH_CM,
        horizontal_aperture=D455_HORIZONTAL_APERTURE_CM,
        clipping_range=(min_range, max_range),
    )

    sensor_noise: SensorNoiseCfg = SensorNoiseCfg(
        enable=True,
        mode="combined",
        depth_std=0.005,
        depth_std_multiplier=0.01,
        dropout_prob=0.01,
        dropout_value=-1.0,
    )


@configclass
class D455CameraCfg(D455Camera, CameraCfg):
    pass


@configclass
class TiledD455CameraCfg(D455Camera, TiledCameraCfg):
    pass

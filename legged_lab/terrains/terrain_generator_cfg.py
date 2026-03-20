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

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.28,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.30,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.32,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.34,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        # "high_platform": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        # ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.15, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

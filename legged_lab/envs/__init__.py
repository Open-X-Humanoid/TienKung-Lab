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

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.dex.dex_env import DexEnv
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)

from legged_lab.envs.dex.dex_env import DexEnv
from legged_lab.envs.dex.run_cfg import DexRunAgentCfg, DexRunFlatEnvCfg
from legged_lab.envs.dex.walk_cfg import DexWalkAgentCfg, DexWalkFlatEnvCfg

from legged_lab.utils.task_registry import task_registry

task_registry.register("lite_walk", TienKungEnv, TienKungWalkFlatEnvCfg, TienKungWalkAgentCfg)
task_registry.register("lite_run", TienKungEnv, TienKungRunFlatEnvCfg, TienKungRunAgentCfg)
# task_registry.register("lite_run", TienKungEnv, TienKungRunFlatEnvCfg, TienKungRunAgentCfg)
# task_registry.register(
#     "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg, TienKungWalkWithSensorAgentCfg
# )
# task_registry.register(
#     "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg, TienKungRunWithSensorAgentCfg
# )
task_registry.register("dex_walk", DexEnv, DexWalkFlatEnvCfg, DexWalkAgentCfg)
task_registry.register("dex_run", DexEnv, DexRunFlatEnvCfg, DexRunAgentCfg)

import numpy as np
import torch
from matplotlib import pyplot as plt

class GridAdaptiveCurriculum:
    def __init__(self, cfg, seed=1):
        self.rng = np.random.default_rng(seed)

        self.cmd_cfg = cfg
        self.keys = sorted(self.cmd_cfg.keys())  #  keys = ["x", "y", "yaw"]

        self.lows = np.array([self.cmd_cfg[key]['limit_low'] for key in self.keys])
        self.highs = np.array([self.cmd_cfg[key]['limit_high'] for key in self.keys])
        self.bin_sizes = np.array([(self.cmd_cfg[key]['limit_high'] - self.cmd_cfg[key]['limit_low']) / self.cmd_cfg[key]['num_bins'] for key in self.keys])
        self.local_ranges = np.array([self.cmd_cfg[key]['local_range'] for key in self.keys])

        self.bin_cmd_values = [
            np.linspace(
                self.cmd_cfg[key]['limit_low'] + self.bin_sizes[i] / 2,
                self.cmd_cfg[key]['limit_high'] - self.bin_sizes[i] / 2,
                self.cmd_cfg[key]['num_bins']
            )
            for i, key in enumerate(self.keys)
        ]

        self.bin_values_grid = np.stack(np.meshgrid(*self.bin_cmd_values, indexing='ij')).reshape(len(self.keys), -1).T
        self.n_combinations = len(self.bin_values_grid)
        self.weights = np.zeros(self.n_combinations)
        self.indices = np.arange(self.n_combinations)

        init_low = np.array([self.cmd_cfg[key]['init_low'] for key in self.keys])
        init_high = np.array([self.cmd_cfg[key]['init_high'] for key in self.keys])

        initial_inds = np.logical_and(
            self.bin_values_grid >= init_low,
            self.bin_values_grid <= init_high
        ).all(axis=1)

        if not initial_inds.any():
            raise ValueError("Empty domain!")

        self.weights[initial_inds] = 1.0

        self.success_num = np.zeros(self.n_combinations)
        self.total_num = np.zeros(self.n_combinations)
    
    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.full(self.bin_values_grid.shape[1], ranges)
        
        adjacent_inds = np.logical_and(
            self.bin_values_grid[:, None, :] >= self.bin_values_grid[bin_inds, :] - ranges,
            self.bin_values_grid[:, None, :] <= self.bin_values_grid[bin_inds, :] + ranges,
        ).all(axis=-1)
        
        return [adjacent_inds[:, i].nonzero()[0] for i in range(adjacent_inds.shape[1])]
    
    def sample_bins(self, batch_size):
        inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())
        return self.bin_values_grid[inds], inds
    
    def sample_uniform_from_bin(self, center):
        low, high = center - self.bin_sizes / 2, center + self.bin_sizes / 2
        return self.rng.uniform(low, high, size=center.shape)
    
    def sample(self, batch_size):
        center, inds = self.sample_bins(batch_size)
        cmd = self.sample_uniform_from_bin(center)
        return cmd, inds 
    
    def update_success_rate(self, bin_inds, task_rewards, success_thresholds):
        is_success = np.all([(task_reward > success_threshold).cpu().numpy() for task_reward, success_threshold in zip(task_rewards, success_thresholds)], axis=0)
        is_success = np.nonzero(is_success)[0]
        
        self.success_num[bin_inds[is_success]] += 1.0
        self.total_num[bin_inds] += 1.0
    
    def update_weights(self):
        success_rate = self.success_num / (self.total_num + 1e-6)
        successful_bins = np.nonzero(success_rate > 0.5)[0]
        
        self.weights[successful_bins] = np.clip(self.weights[successful_bins] + 0.2, 0, 1)
        
        for adjacent in self.get_local_bins(successful_bins, ranges=self.local_ranges):
            self.weights[adjacent] = np.clip(self.weights[adjacent] + 0.2, 0, 1)
        
        self.success_num.fill(0)
        self.total_num.fill(0)


if __name__ == '__main__':
    cmd_cfg = {
        "x": {
            'init_low': -0.5,
            'init_high': 0.5,
            'limit_low': -1.6,
            'limit_high': 1.6,
            'local_range': 0.2,
            'num_bins': 21,
        },
        "y": {
            'init_low': -0.3,
            'init_high': 0.3,
            'limit_low': -0.6,
            'limit_high': 0.6,
            'local_range': 0.2,
            'num_bins': 11,
        },
        "yaw": {
            'init_low': -0.5,
            'init_high': 0.5,
            'limit_low': -0.7,
            'limit_high': 0.7,
            'local_range': 0.2,
            'num_bins': 11,
        }
    }

    curriculum = GridAdaptiveCurriculum(cmd_cfg)
    
    cmds, bin_indices = curriculum.sample(5)
    print("Sampled commands:", cmds)
    print("Sampled bin indices:", bin_indices)
    
    task_rewards = [torch.tensor(np.random.uniform(0, 1, len(bin_indices))) for _ in range(3)]
    success_thresholds = [0.5, 0.5, 0.5]
    curriculum.update_success_rate(bin_indices, task_rewards, success_thresholds)

    curriculum.update_weights()
    print("Updated weights:", curriculum.weights)
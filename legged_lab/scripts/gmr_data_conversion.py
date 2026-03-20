import pickle
import numpy as np
import torch
import argparse
from scipy.spatial.transform import Rotation

# ================== 自定义四元数操作函数 ==================
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    四元数乘法
    q1, q2: (..., 4) 或 (N, 4) 张量，格式为 (w, x, y, z)
    返回: q1 * q2
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    四元数共轭
    q: (..., 4) 张量，格式为 (w, x, y, z)
    返回: q* 共轭
    """
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)

def axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """
    从四元数提取轴角表示
    q: (..., 4) 张量，格式为 (w, x, y, z)
    返回: (..., 3) 轴角向量 (axis * angle)
    """
    # 确保四元数已归一化
    q_norm = q / torch.norm(q, dim=-1, keepdim=True)
    
    # 提取角度
    angles = 2.0 * torch.acos(torch.clamp(q_norm[..., 0], -1.0, 1.0))
    
    # 提取轴
    sin_half_angle = torch.sqrt(torch.clamp(1.0 - q_norm[..., 0] * q_norm[..., 0], 0.0, 1.0))
    
    # 避免除以零
    mask = sin_half_angle > 1e-7
    axes = torch.zeros_like(q_norm[..., 1:])
    axes[mask] = q_norm[..., 1:][mask] / sin_half_angle[mask].unsqueeze(-1)
    
    return axes * angles.unsqueeze(-1)

# ================== 主转换函数 ==================
def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])         
    dq = quat_mul(q1_conj, root_rot_t[1:])            
    axis_angle = axis_angle_from_quat(dq)             
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)
    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1],  
         root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps)
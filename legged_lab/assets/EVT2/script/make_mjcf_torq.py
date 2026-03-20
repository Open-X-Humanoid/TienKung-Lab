from urdf2mjcf.convert import convert_urdf_to_mjcf
from urdf2mjcf.model import ActuatorMetadata, JointMetadata
from pathlib import Path
import re
import math
import xml.etree.ElementTree as ET

def extract_joint_limits(urdf_path):
    """从 URDF 文件中提取关节名称和对应的限位信息（effort, lower, upper）"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    joint_info = {}
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        if joint_name is None:
            continue
        
        limit = joint.find('limit')
        if limit is not None:
            info = {}
            if limit.get('effort') is not None:
                info['effort'] = float(limit.get('effort'))
            if limit.get('lower') is not None:
                info['lower'] = float(limit.get('lower'))  # 弧度
            if limit.get('upper') is not None:
                info['upper'] = float(limit.get('upper'))  # 弧度
            if info:
                joint_info[joint_name] = info
    
    return joint_info

def create_metadata(urdf_path):
    """创建关节和执行器的元数据"""
    # 提取关节限位信息（effort, lower, upper）
    joint_info = extract_joint_limits(urdf_path)
    print(f"Extracted limit information for {len(joint_info)} joints from URDF")
    
    # 解析 URDF 获取所有关节
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 创建关节元数据
    joint_metadata = {}
    actuator_metadata = {}
    actuator_type_to_joints = {}  # 用于跟踪每个 actuator_type 对应的关节
    
    joint_idx = 0
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        if joint_name is None:
            continue
        
        # 跳过 fixed 关节
        if joint.get('type') == 'fixed':
            continue
        
        info = joint_info.get(joint_name, {})
        effort = info.get('effort')
        lower_rad = info.get('lower')
        upper_rad = info.get('upper')
        
        # 为每个不同的 effort 值创建唯一的 actuator_type
        if effort is not None:
            # 创建 actuator_type 名称（基于 effort 值，使用整数部分避免浮点精度问题）
            # 例如：332.0 -> motor_332, 6.3 -> motor_6_3
            if effort == int(effort):
                actuator_type = f"motor_{int(effort)}"
            else:
                actuator_type = f"motor_{effort}".replace('.', '_')
            
            # 根据 effort 值设置阻尼和摩擦力
            # damping: 速度阻尼系数 (N·m·s/rad)，与速度成正比
            # frictionloss: 库伦摩擦损失 (N·m)，恒定摩擦力
            if effort >= 200:  # 大功率关节
                damping = 2.0  # 较大的速度阻尼
                frictionloss = 1.0  # 较大的库伦摩擦
            elif effort >= 50:  # 中等功率关节
                damping = 1.0
                frictionloss = 0.5
            else:  # 小功率关节
                damping = 0.5
                frictionloss = 0.1
            
            # 如果这个 actuator_type 还不存在，创建它
            if actuator_type not in actuator_metadata:
                actuator_metadata[actuator_type] = ActuatorMetadata(
                    actuator_type=actuator_type,
                    max_torque=effort,
                    damping=damping,
                    frictionloss=frictionloss
                )
                print(f"Created actuator type '{actuator_type}' with max_torque={effort} N·m, damping={damping}, frictionloss={frictionloss}")
        else:
            # 如果没有 effort 值，使用默认的 motor
            actuator_type = "motor"
            if actuator_type not in actuator_metadata:
                actuator_metadata[actuator_type] = ActuatorMetadata(
                    actuator_type=actuator_type,
                    damping=0.5,  # 默认阻尼
                    frictionloss=0.1  # 默认摩擦力
                )
                print(f"Created default actuator type '{actuator_type}' with damping=0.5, frictionloss=0.1")
        
        # 将弧度转换为度（用于 JointMetadata）
        min_angle_deg = None
        max_angle_deg = None
        if lower_rad is not None:
            min_angle_deg = math.degrees(lower_rad)
        if upper_rad is not None:
            max_angle_deg = math.degrees(upper_rad)
        
        # 创建关节元数据
        joint_metadata[joint_name] = JointMetadata(
            actuator_type=actuator_type,
            id=joint_idx,
            nn_id=joint_idx,
            kp=1.0,
            kd=1.0,
            soft_torque_limit=effort if effort is not None else 1.0,
            min_angle_deg=min_angle_deg,
            max_angle_deg=max_angle_deg,
        )
        
        limit_str = ""
        if lower_rad is not None and upper_rad is not None:
            limit_str = f" range=[{lower_rad:.3f}, {upper_rad:.3f}] rad"
        elif lower_rad is not None:
            limit_str = f" lower={lower_rad:.3f} rad"
        elif upper_rad is not None:
            limit_str = f" upper={upper_rad:.3f} rad"
        if limit_str:
            print(f"  Joint {joint_name}:{limit_str}")
        
        # 跟踪每个 actuator_type 对应的关节
        if actuator_type not in actuator_type_to_joints:
            actuator_type_to_joints[actuator_type] = []
        actuator_type_to_joints[actuator_type].append(joint_name)
        
        joint_idx += 1
    
    # 打印总结
    print(f"\nActuator type summary:")
    for actuator_type, joints in actuator_type_to_joints.items():
        effort_val = actuator_metadata[actuator_type].max_torque
        damping_val = actuator_metadata[actuator_type].damping
        frictionloss_val = actuator_metadata[actuator_type].frictionloss
        effort_str = f" (max_torque={effort_val} N·m)" if effort_val is not None else ""
        damping_str = f", damping={damping_val}" if damping_val is not None else ""
        frictionloss_str = f", frictionloss={frictionloss_val}" if frictionloss_val is not None else ""
        print(f"  {actuator_type}{effort_str}{damping_str}{frictionloss_str}: {len(joints)} joints")
    
    return joint_metadata, actuator_metadata

def main(urdf_path, mjcf_file):
    # 创建元数据
    joint_metadata, actuator_metadata = create_metadata(urdf_path)
    
    print(f"\nConverting URDF to MJCF with {len(joint_metadata)} joints and {len(actuator_metadata)} actuator types")
    
    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=mjcf_file,
        joint_metadata=joint_metadata,
        actuator_metadata=actuator_metadata
    )
    
    print(f"MJCF file created at {mjcf_file}")

    with open(mjcf_file, "r", encoding="utf-8") as f:
        mjcf_content = f.read()
    
    # 替换 package:// 路径
    mjcf_content = re.sub(r'package://[^/]+/', './', mjcf_content)
    
    # 添加地面到 worldbody
    ground_geom = '    <!-- 地面 -->\n    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" quat="1 0 0 0" rgba="0.8 0.8 0.8 1" friction="1 0.005 0.0001" condim="3" />\n    \n'
    
    # 在 <worldbody> 标签后添加地面
    if '<worldbody>' in mjcf_content:
        mjcf_content = re.sub(
            r'(<worldbody>\s*\n)',
            r'\1' + ground_geom,
            mjcf_content,
            count=1
        )
        print("Ground added to MJCF file")
    else:
        print("Warning: <worldbody> tag not found, ground not added")
    
    
    # 将 <material name="collision_material" ... /> 的颜色改为灰色
    def replace_collision_color(match):
        return match.group(1) + '0.5 0.5 0.5 0.9' + match.group(2)
    
    mjcf_content = re.sub(
        r'(<material\s+name="collision_material"\s+rgba=")[^"]*(")',
        replace_collision_color,
        mjcf_content
    )
    with open(mjcf_file, "w", encoding="utf-8") as f:
        f.write(mjcf_content)

if __name__ == "__main__":
    urdf_path = Path("../urdf/tiangong2dex.urdf").resolve()
    mjcf_file = Path("../tiangong2dex_torq.xml").resolve()
    

    main(urdf_path, mjcf_file)


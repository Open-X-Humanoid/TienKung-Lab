# 导入兼容低版本Python的类型注解
from typing import List

class IsaacJointOrderConverter:
    """用于Isaac Lab和Isaac Gym关节顺序互相转换的工具类"""

    def __init__(self, gym_joint_names: List[str], lab_joint_names: List[str]):
        """
        构造函数：初始化关节名映射表（需根据机器人型号自定义）
        Args:
            gym_joint_names: Isaac Gym的关节名列表（原始顺序）
            lab_joint_names: Isaac Lab的关节名列表（标准化顺序）
        """
        self.gym_joint_names = gym_joint_names
        self.lab_joint_names = lab_joint_names

        # 存储索引映射的字典（key: 源索引，value: 目标索引）
        self.gym2lab_idx_dict = {}
        self.lab2gym_idx_dict = {}

        # 建立关节名到索引的映射（方便快速查找）
        gym_name_to_idx = {name: idx for idx, name in enumerate(gym_joint_names)}
        lab_name_to_idx = {name: idx for idx, name in enumerate(lab_joint_names)}

        # 遍历Lab关节名，匹配Gym关节名并建立索引映射（直接完全匹配，不去除后缀）
        for lab_idx, lab_name in enumerate(lab_joint_names):
            # 直接使用原始名称匹配，不再处理_joint后缀
            if lab_name in gym_name_to_idx:
                gym_idx = gym_name_to_idx[lab_name]
                self.gym2lab_idx_dict[gym_idx] = lab_idx
                self.lab2gym_idx_dict[lab_idx] = gym_idx

        # 遍历Gym关节名，确保所有Gym关节都能匹配到Lab（可选，根据需求调整）
        # 如果你只需要匹配Lab中存在的Gym关节，可注释以下校验，保留上面的逻辑
        # 校验映射完整性（注意：若两者关节名数量/名称不一致，会触发异常）
        if len(self.gym2lab_idx_dict) != len(gym_joint_names) or len(self.lab2gym_idx_dict) != len(lab_joint_names):
            # 打印未匹配的关节名，方便调试
            unmatch_gym = [name for idx, name in enumerate(gym_joint_names) if idx not in self.gym2lab_idx_dict]
            unmatch_lab = [name for idx, name in enumerate(lab_joint_names) if idx not in self.lab2gym_idx_dict]
            print(f"警告：未匹配的Gym关节名：{unmatch_gym}")
            print(f"警告：未匹配的Lab关节名：{unmatch_lab}")
            # 若需要严格匹配，保留raise；若只需要部分匹配，注释raise
            # raise RuntimeError("关节名映射不完整，请检查关节名是否匹配！")

        # 初始化一维索引映射数组（提前构建，避免重复计算）
        self.gym_to_lab_idx = self._init_index_array(self.gym2lab_idx_dict, len(gym_joint_names))
        self.lab_to_gym_idx = self._init_index_array(self.lab2gym_idx_dict, len(lab_joint_names))

    def _init_index_array(self, idx_dict: dict, length: int) -> List[int]:
        """
        初始化一维索引映射数组
        Args:
            idx_dict: 索引映射字典（key: 源索引，value: 目标索引）
            length: 数组长度
        Returns:
            一维索引映射数组（未匹配的索引值设为-1，方便识别）
        """
        index_array = [-1] * length  # 未匹配的索引设为-1，替代原来的0
        for src_idx, dst_idx in idx_dict.items():
            index_array[src_idx] = dst_idx
        return index_array

    def gym_to_lab(self, gym_data: List[float]) -> List[float]:
        """
        Isaac Gym关节数据 → Isaac Lab关节数据（按顺序转换）
        Args:
            gym_data: Gym的关节数据（如角度、力矩，顺序与gym_joint_names一致）
        Returns:
            Lab的关节数据（顺序与lab_joint_names一致，未匹配的设为0.0）
        """
        if len(gym_data) != len(self.gym_joint_names):
            raise ValueError("Gym数据长度与关节数不匹配！")

        lab_data = [0.0] * len(self.lab_joint_names)
        for gym_idx, lab_idx in self.gym2lab_idx_dict.items():
            lab_data[lab_idx] = gym_data[gym_idx]
        return lab_data

    def lab_to_gym(self, lab_data: List[float]) -> List[float]:
        """
        Isaac Lab关节数据 → Isaac Gym关节数据（按顺序转换）
        Args:
            lab_data: Lab的关节数据（顺序与lab_joint_names一致）
        Returns:
            Gym的关节数据（顺序与gym_joint_names一致，未匹配的设为0.0）
        """
        if len(lab_data) != len(self.lab_joint_names):
            raise ValueError("Lab数据长度与关节数不匹配！")

        gym_data = [0.0] * len(self.gym_joint_names)
        for lab_idx, gym_idx in self.lab2gym_idx_dict.items():
            gym_data[gym_idx] = lab_data[lab_idx]
        return gym_data

    def print_gym_to_lab_index_array(self):
        """打印输出Gym→Lab的索引映射一维数组（与示例格式一致）"""
        print("isaac_to_mujoco_idx = [")
        self._print_array_content(self.gym_to_lab_idx)
        print("]")

    def print_lab_to_gym_index_array(self):
        """打印输出Lab→Gym的索引映射一维数组（与示例格式一致）"""
        print("\nmujoco_to_isaac_idx = [")
        self._print_array_content(self.lab_to_gym_idx)
        print("]")

    def _print_array_content(self, arr: List[int]):
        """
        打印数组内容（严格匹配示例的格式：每行10个元素、缩进、逗号）
        Args:
            arr: 要打印的索引数组
        """
        items_per_line = 10  # 每行显示10个元素，与示例一致
        count = 0
        # 遍历数组元素
        for i, num in enumerate(arr):
            # 每行开头添加2个空格缩进
            if count == 0:
                print("  ", end="")
            # 打印元素
            print(num, end="")
            # 不是最后一个元素则加逗号和空格
            if i != len(arr) - 1:
                print(", ", end="")
            count += 1
            # 每10个元素换行，重置计数（最后一行除外）
            if count == items_per_line and i != len(arr) - 1:
                print()
                count = 0
        # 最后一行换行
        print()


# 测试示例：使用你提供的关节名列表
if __name__ == "__main__":
    try:
        # ********************* 你提供的关节名列表 *********************
        # Isaac Gym的关节名（包含_joint后缀）
        gym_joint_names_dex = [
            "hip_pitch_l_joint",
            "hip_roll_l_joint",
            "hip_yaw_l_joint",
            "knee_pitch_l_joint",
            "ankle_pitch_l_joint",
            "ankle_roll_l_joint",
            "hip_pitch_r_joint",
            "hip_roll_r_joint",
            "hip_yaw_r_joint",
            "knee_pitch_r_joint",
            "ankle_pitch_r_joint",
            "ankle_roll_r_joint",

            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",

            "shoulder_pitch_l_joint",
            "shoulder_roll_l_joint",
            "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint",
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
        ]
        
        gym_joint_names_lite = [
            "hip_roll_l_joint",
            "hip_pitch_l_joint",
            "hip_yaw_l_joint",
            "knee_pitch_l_joint",
            "ankle_pitch_l_joint",
            "ankle_roll_l_joint",
            "hip_roll_r_joint",
            "hip_pitch_r_joint",
            "hip_yaw_r_joint",
            "knee_pitch_r_joint",
            "ankle_pitch_r_joint",
            "ankle_roll_r_joint",

            "shoulder_pitch_l_joint",
            "shoulder_roll_l_joint",
            "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint",
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
        ]
        
        gym_joint_names_pro = [
            "hip_roll_l_joint",
            "hip_pitch_l_joint",
            "hip_yaw_l_joint",
            "knee_pitch_l_joint",
            "ankle_pitch_l_joint",
            "ankle_roll_l_joint",
            "hip_roll_r_joint",
            "hip_pitch_r_joint",
            "hip_yaw_r_joint",
            "knee_pitch_r_joint",
            "ankle_pitch_r_joint",
            "ankle_roll_r_joint",

            "waist_yaw_joint",

            "shoulder_pitch_l_joint",
            "shoulder_roll_l_joint",
            "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint",
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
        ]
                
        # Isaac Lab的关节名（包含_joint后缀，顺序与Gym不同）
        lab_joint_names_dex = [
            'hip_pitch_l_joint', 'hip_pitch_r_joint', 'waist_yaw_joint', 'hip_roll_l_joint', 'hip_roll_r_joint', 'waist_roll_joint', 
            'hip_yaw_l_joint', 'hip_yaw_r_joint', 'waist_pitch_joint', 'knee_pitch_l_joint', 'knee_pitch_r_joint', 'shoulder_pitch_l_joint',
            'shoulder_pitch_r_joint', 'ankle_pitch_l_joint', 'ankle_pitch_r_joint', 'shoulder_roll_l_joint', 'shoulder_roll_r_joint', 'ankle_roll_l_joint', 
            'ankle_roll_r_joint', 'shoulder_yaw_l_joint', 'shoulder_yaw_r_joint', 'elbow_pitch_l_joint', 'elbow_pitch_r_joint'
        ]
        
                # Isaac Lab的关节名（包含_joint后缀，顺序与Gym不同）
        lab_joint_names_lite = ['hip_roll_l_joint', 'hip_roll_r_joint', 'shoulder_pitch_l_joint', 'shoulder_pitch_r_joint', 'hip_pitch_l_joint', 'hip_pitch_r_joint', 
                            'shoulder_roll_l_joint', 'shoulder_roll_r_joint', 'hip_yaw_l_joint', 'hip_yaw_r_joint', 'shoulder_yaw_l_joint', 'shoulder_yaw_r_joint',
                            'knee_pitch_l_joint', 'knee_pitch_r_joint', 'elbow_pitch_l_joint', 'elbow_pitch_r_joint', 'ankle_pitch_l_joint', 'ankle_pitch_r_joint', 'ankle_roll_l_joint', 'ankle_roll_r_joint']

        
        lab_joint_names_pro= ['hip_roll_l_joint', 'hip_roll_r_joint', 'waist_yaw_joint', 'hip_pitch_l_joint', 'hip_pitch_r_joint', 
                            'shoulder_pitch_l_joint', 'shoulder_pitch_r_joint', 'hip_yaw_l_joint', 'hip_yaw_r_joint', 'shoulder_roll_l_joint', 'shoulder_roll_r_joint',
                            'knee_pitch_l_joint', 'knee_pitch_r_joint', 'shoulder_yaw_l_joint', 'shoulder_yaw_r_joint', 'ankle_pitch_l_joint', 'ankle_pitch_r_joint',  'elbow_pitch_l_joint', 'elbow_pitch_r_joint', 'ankle_roll_l_joint', 'ankle_roll_r_joint', 
                            ]
        
        # ******************************************************************

        # 创建转换器
        converter = IsaacJointOrderConverter(gym_joint_names_pro, lab_joint_names_pro)

        # 测试数据转换：调整数据长度与Gym关节数一致（23个）
        gym_joint_angles = [i * 0.1 for i in range(len(gym_joint_names_pro))]
        lab_joint_angles = converter.gym_to_lab(gym_joint_angles)
        print("Gym → Lab 转换后的关节角度：")
        print(" ".join([f"{angle:.1f}" for angle in lab_joint_angles]))
        print()

        # 核心：打印输出一维索引映射数组
        converter.print_gym_to_lab_index_array()
        converter.print_lab_to_gym_index_array()

    except Exception as e:
        print(f"错误：{e}")

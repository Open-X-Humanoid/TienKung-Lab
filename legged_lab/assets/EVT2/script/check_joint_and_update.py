import csv
import sys
import xml.etree.ElementTree as ET
import math
def parse_urdf_joints(urdf_path):
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints = []
        for joint in root.findall('joint'):
            name = joint.get('name')
            if name:
                joints.append(name)
        return joints
    except Exception as e:
        print(f"Error parsing URDF file: {e}")
        return []



def find_joint_name_in_csv(csv_path):
    print(f"读取 {csv_path} 中的 joint 命名...")
    joint_names = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        # 跳过所有行直到找到"joint部分"
        found_joint_section = False
        for row in reader:
            # 如果遇到"joint部分"行，开始读取
            if len(row) > 0 and row[0] == 'joint部分':
                found_joint_section = True
                continue  # 跳过标题行
            
            # 只在找到joint部分后读取
            if not found_joint_section:
                continue
            
            # 跳过空行
            if not row or len(row) < 2:
                continue
            
            # 提取joint命名（第二列，索引为1）
            joint_name = row[1].strip()
            
            # 只添加非空的joint名称
            if joint_name and joint_name != '':
                joint_names.append(joint_name)
    
    # print(f"\n关键参数表中找到 {len(joint_names)} 个 joint:")
    # for name in joint_names:
    #     print(f"  - {name}")
    return joint_names

def check_urdf_joints(urdf_file, expected_joints, verbose=True):
    """
    检查URDF文件中是否包含所有预期的joint名。
    :param urdf_file: URDF文件路径
    :param expected_joints: 预期的joint名称列表
    :param verbose: 是否打印详细信息
    :return: missing_joints列表，如果没有缺失则为空
    """
    joints = parse_urdf_joints(urdf_file)
    missing_joints = [name for name in expected_joints if name not in joints]
    if verbose:
        if missing_joints:
            print("Missing joints in URDF:")
            for name in missing_joints:
                print(f"  {name}")
        else:
            print("All expected joints are present in the URDF.")
    return missing_joints

def read_joint_properties_in_csv(csv_path, joint_name):
    joint_row_num = None
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        found_joint_section = False
        for row in reader:
            # 寻找 joint 部分开始
            if not found_joint_section:
                if len(row) > 0 and row[0] == 'joint部分':
                    found_joint_section = True
                continue
            # 跳过空行及非参数行
            if not row or len(row) < 2:
                continue

            if row[1].strip() == joint_name:
                joint_type = row[2].strip()
                joint_axis = row[3].strip()
                joint_lower = row[4].strip()
                joint_upper = row[5].strip()
                joint_velocity = row[6].strip()
                joint_effort = row[7].strip()
                
                if joint_lower == '' or joint_lower == 'null':
                    joint_lower = 0.0
                else:
                    joint_lower = float(joint_lower)
                if joint_upper == '' or joint_upper == 'null':
                    joint_upper = 0.0
                else:
                    joint_upper = float(joint_upper)
                if joint_effort == '' or joint_effort == 'null':
                    joint_effort = 0.0
                else:
                    joint_effort = float(joint_effort)
                if joint_velocity == '' or joint_velocity == 'null':
                    joint_velocity = 0.0
                else:
                    joint_velocity = float(joint_velocity)

                # 单位转换
                joint_lower = joint_lower * math.pi / 180
                joint_upper = joint_upper * math.pi / 180
                # 将每分钟转为弧度每秒
                joint_velocity = joint_velocity * 2 * math.pi / 60


                joint_properties = {
                    "joint_type": joint_type,
                    "joint_axis": joint_axis,
                    "joint_lower": joint_lower,
                    "joint_upper": joint_upper,
                    "joint_velocity": joint_velocity,
                    "joint_effort": joint_effort,
                }   
                return joint_properties
    return None

def update_urdf_joint_properties(urdf_file, joint_name, joint_properties):
    if joint_properties is None:
        return
    if joint_properties['joint_type'] == 'fixed':
        return

    tree = ET.parse(urdf_file)
    root = tree.getroot()
    for joint in root.findall('joint'):
        if joint.get('name') == joint_name:
            # 读取joint的axis
            axis_el = joint.find('axis')
            joint_axis = None
            if axis_el is not None and 'xyz' in axis_el.attrib:
                joint_axis = axis_el.attrib['xyz']
                if joint_properties['joint_axis'] == 'x':
                    if joint_axis != '1 0 0':
                        axis_el.set('xyz', '1 0 0')
                        print("update joint:", joint_name, "axis:", joint_axis, "to:", '1 0 0')
                elif joint_properties['joint_axis'] == 'y':
                    if joint_axis != '0 1 0':
                        axis_el.set('xyz', '0 1 0')
                        print("update joint:", joint_name, "axis:", joint_axis, "to:", '0 1 0')
                elif joint_properties['joint_axis'] == 'z':
                    if joint_axis != '0 0 1':
                        axis_el.set('xyz', '0 0 1')
                        print("update joint:", joint_name, "axis:", joint_axis, "to:", '0 0 1')
            limit = joint.find('limit')
            if limit is not None:
                print("update joint:", joint_name)    
                limit.set('lower', str(joint_properties['joint_lower']))
                limit.set('upper', str(joint_properties['joint_upper']))
                limit.set('velocity', str(joint_properties['joint_velocity']))
                limit.set('effort', str(joint_properties['joint_effort']))  
            tree.write(urdf_file, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    urdf_file = "../urdf/tiangong2dex.urdf"
    csv_file = "../关键参数/URDF关键参数表 - 天工dex EVT2.csv"
    names = find_joint_name_in_csv(csv_file)
    missing_joints = check_urdf_joints(urdf_file, names)

    for joint_name in names:
        if joint_name not in missing_joints:
            joint_properties = read_joint_properties_in_csv(csv_file, joint_name)
            update_urdf_joint_properties(urdf_file, joint_name, joint_properties)



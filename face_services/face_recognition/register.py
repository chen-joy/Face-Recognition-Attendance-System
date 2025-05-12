import face_recognition
import os
import json
import numpy as np
import sys

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_services.tools import generate_id_name_mapping

# 生成ID到姓名的映射
print("生成学号到姓名的映射...")
id_name_mapping = generate_id_name_mapping(source_folder='./members', save_to_all_services=True)

if not id_name_mapping:
    print("错误: 无法生成ID到姓名的映射，请检查members文件夹")
    sys.exit(1)

folder_path = './members'

class_list = []
name_list = []
id_list = []

print("\n使用face_recognition提取特征...")
success_count = 0
fail_count = 0

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(('.png', '.jpg')):
        continue  
    
    try:
        # 解析学号和姓名
        name_parts = os.path.splitext(filename)[0].split('-')
        if len(name_parts) != 2:
            print(f"警告: 文件名格式不正确: {filename}，跳过")
            continue
            
        student_id = name_parts[0].strip()
        name = name_parts[1].strip()
        
        # 构建完整路径
        image_path = os.path.join(folder_path, filename)
        
        # 提取特征
        print(f"处理: {filename} (ID: {student_id}, 姓名: {name})")
        current_image = face_recognition.load_image_file(image_path)
        current_encoding = face_recognition.face_encodings(current_image)[0]
        
        # 保存结果
        name_list.append(filename)
        class_list.append(current_encoding)
        id_list.append(student_id)
        
        success_count += 1
        print(f"特征提取成功: {name}({student_id})")
        
    except Exception as e:
        fail_count += 1
        print(f"特征提取失败 ({filename}): {e}")

print(f"\n特征提取完成: 成功 {success_count}, 失败 {fail_count}")

if success_count == 0:
    print("没有成功提取任何特征，退出")
    sys.exit(1)

# 保存特征和ID映射
try:
    # 将NumPy数组转换为列表以便JSON序列化
    class_list_json = [arr.tolist() for arr in class_list]
    
    # 保存特征列表
    with open("./face_services/face_recognition/class_list.json", "w") as file:
        json.dump(class_list_json, file)
    
    # 保存文件名列表
    with open("./face_services/face_recognition/name_list.json", "w") as file:
        json.dump(name_list, file)
    
    # 保存ID列表
    with open("./face_services/face_recognition/id_list.json", "w") as file:
        json.dump(id_list, file)
    
    print("\n已保存特征和ID映射到face_recognition目录")
    print("注册完成!")
except Exception as e:
    print(f"保存数据时出错: {e}")

"""
集成各人脸识别服务的工具模块
提供统一的导入接口，便于主应用调用不同的识别服务
"""

# 各种人脸识别服务
from .baidu.api import search_baidu
from .face_recognition.api import search_face_recognition
from .facenet.api import search_facenet
from .hyp_oc.api import search_hyp_oc

# 人脸属性分析服务
from .modelscope.api import gender_age_expression_live

import os
import json
import shutil
import base64
import uuid
from datetime import datetime

def standardize_response(recognition_result, confidence=None, student_id=None, name=None, gender=None, age=None, emotion=None, live_score=None, method=None):
    """
    标准化人脸识别结果响应格式
    
    参数:
        recognition_result: 原始识别结果
        confidence: 置信度
        student_id: 学生ID
        name: 姓名
        gender: 性别
        age: 年龄
        emotion: 情绪
        live_score: 活体检测得分
        method: 使用的识别方法
    
    返回:
        标准化的响应字典
    """
    response = {
        "success": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "recognition": {
                "student_id": student_id or "",
                "name": name or "",
                "confidence": confidence or 0.0,
                "method": method or "unknown"
            },
            "attributes": {
                "gender": gender or "",
                "age": age or 0,
                "emotion": emotion or "",
                "live_score": live_score or 0.0
            }
        },
        "raw_result": recognition_result
    }
    return response

def save_base64_image(base64_str, output_dir="./saved_images", prefix="img"):
    """
    保存Base64编码的图像到文件
    
    参数:
        base64_str: Base64编码的图像字符串
        output_dir: 输出目录路径
        prefix: 文件名前缀
    
    返回:
        保存的图像文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 移除可能存在的Base64头部
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    try:
        # 解码Base64数据
        image_data = base64.b64decode(base64_str)
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # 写入文件
        with open(filepath, "wb") as f:
            f.write(image_data)
            
        print(f"图像已保存至: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"保存Base64图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_id_name_mapping(source_folder='./members', save_to_all_services=True):
    """
    从图片文件名生成学号到姓名的映射表
    
    参数:
        source_folder: 包含照片的文件夹路径，默认为./members
        save_to_all_services: 是否将映射保存到所有服务目录
        
    返回:
        映射字典: {学号: 姓名}
    """
    if not os.path.exists(source_folder):
        print(f"错误: 文件夹 {source_folder} 不存在")
        return {}
    
    mapping = {}
    
    # 遍历文件夹中的所有图片
    for filename in os.listdir(source_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # 尝试解析文件名（格式: 学号-姓名.扩展名）
        try:
            name_parts = os.path.splitext(filename)[0].split('-')
            if len(name_parts) == 2:
                student_id = name_parts[0].strip()
                name = name_parts[1].strip()
                mapping[student_id] = name
                print(f"已解析: ID={student_id}, 姓名={name}")
            else:
                print(f"警告: 文件名格式不正确: {filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
    
    # 保存到指定位置
    if mapping:
        # 基础目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 服务列表
        services = {
            'baidu': os.path.join(base_dir, 'face_services', 'baidu', 'id_to_name_mapping.json'),
            'face_recognition': os.path.join(base_dir, 'face_services', 'face_recognition', 'id_to_name_mapping.json'),
            'facenet': os.path.join(base_dir, 'face_services', 'facenet', 'id_to_name_mapping.json'),
            'hyp_oc': os.path.join(base_dir, 'face_services', 'hyp_oc', 'id_to_name_mapping.json')
        }
        
        if save_to_all_services:
            # 保存到所有服务
            for service, filepath in services.items():
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=4)
                print(f"已保存ID到姓名映射到: {filepath}")
        else:
            # 只保存到默认位置
            default_path = os.path.join(base_dir, 'id_to_name_mapping.json')
            with open(default_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=4)
            print(f"已保存ID到姓名映射到: {default_path}")
    
    return mapping

def update_student_database_from_members(folder_path='./members'):
    """
    从members文件夹更新hyp_oc的学生数据库
    
    参数:
        folder_path: members文件夹路径
    """
    mapping = generate_id_name_mapping(folder_path, save_to_all_services=False)
    
    if not mapping:
        print("没有找到有效的ID-姓名映射，无法更新数据库")
        return False
        
    # 读取现有的学生数据库
    student_db_path = './face_services/hyp_oc/student_database.json'
    
    try:
        if os.path.exists(student_db_path):
            with open(student_db_path, 'r', encoding='utf-8') as f:
                student_db = json.load(f)
        else:
            # 创建新的数据库结构
            student_db = {"students": {}, "attendance": {}}
            
        # 更新学生信息
        import datetime
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for student_id, name in mapping.items():
            # 如果学生ID不在数据库中，添加它
            if student_id not in student_db.get("students", {}):
                student_db.setdefault("students", {})[student_id] = {
                    "name": name,
                    "register_time": now
                }
                print(f"添加新学生: {name}({student_id})")
            else:
                # 更新现有学生的姓名（如果不同）
                if student_db["students"][student_id]["name"] != name:
                    student_db["students"][student_id]["name"] = name
                    print(f"更新学生姓名: {student_id} -> {name}")
        
        # 保存更新后的数据库
        with open(student_db_path, 'w', encoding='utf-8') as f:
            json.dump(student_db, f, ensure_ascii=False, indent=4)
            
        print(f"成功更新学生数据库: {student_db_path}")
        return True
        
    except Exception as e:
        print(f"更新学生数据库时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 当作为独立脚本运行时，生成映射并保存到所有服务
    print("从members文件夹生成ID到姓名映射...")
    generate_id_name_mapping()
    print("\n更新hyp_oc学生数据库...")
    update_student_database_from_members()
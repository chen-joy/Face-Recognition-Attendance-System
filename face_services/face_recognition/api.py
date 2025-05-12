import json
import face_recognition
import numpy as np
import traceback
import os

# 定义用于获取绝对路径的辅助函数
def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容不同的运行环境"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

def search_face_recognition(img_path):
    """
    使用face_recognition库进行人脸识别
    
    参数:
        img_path: 图像路径
        
    返回:
        response: 包含识别结果的字典
    """
    response = {}
    
    try:
        # 加载特征库和姓名列表
        try:
            class_list_path = get_resource_path("class_list.json")
            name_list_path = get_resource_path("name_list.json")
            
            with open(class_list_path, "r") as file:
                class_list = json.load(file)

            with open(name_list_path, "r") as file:
                name_list = json.load(file)
                
            print(f"已加载特征库: {len(class_list)}个特征, {len(name_list)}个名称")
        except Exception as e:
            print(f"加载特征库或姓名列表失败: {e}")
            traceback.print_exc()
            response['user_id'] = "None"
            response['score'] = 0.0
            return response
            
        # 从图像中提取特征
        try:
            unknown_image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(unknown_image)
            
            # 检查是否检测到人脸
            if not face_locations:
                print("未检测到人脸")
                response['user_id'] = "None"
                response['score'] = 0.0
                return response
                
            unknown_encoding = face_recognition.face_encodings(unknown_image, face_locations)[0]
        except Exception as e:
            print(f"人脸特征提取失败: {e}")
            response['user_id'] = "None"
            response['score'] = 0.0
            return response
        
        # 进行人脸比对
        distances = face_recognition.face_distance(class_list, unknown_encoding)
        best_match_index = np.argmin(distances)
        match_distance = distances[best_match_index]
        
        # 计算相似度分数 (距离越小，相似度越高)
        similarity_score = 1 - match_distance
        
        # 如果相似度大于阈值，认为匹配成功
        threshold = 0.5  # 降低阈值，提高匹配成功率 (原值0.6)
        if similarity_score >= threshold:
            name = name_list[best_match_index]
            print(f"识别为：{name}, 相似度: {similarity_score:.4f}")
            
            id_name = name.split('.')[0]  # 去除文件后缀
            
            # 尝试分割学号和姓名
            if '-' in id_name:
                id_number, real_name = id_name.split('-')  # 根据'-'分割学号和姓名
                response['user_id'] = real_name
            else:
                response['user_id'] = id_name
                
            response['score'] = float(similarity_score)
        else:
            print(f"未找到匹配. 最佳相似度: {similarity_score:.4f}")
            response['user_id'] = "None"
            response['score'] = float(similarity_score)
    
    except Exception as e:
        print(f"Face Recognition处理出错: {e}")
        traceback.print_exc()
        response['user_id'] = "None"
        response['score'] = 0.0
    
    return response


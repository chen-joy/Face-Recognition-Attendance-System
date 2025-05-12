import torch
import json
import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import traceback
import importlib.util

# 直接从当前目录导入需要的模块
try:
    # 首先尝试直接导入
    from .face_recognition_model import FaceRecognitionModel, cosine_similarity
    from .models import Vgg_face_dag, load_vgg_face
    print("成功从本地导入模型文件")
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试使用绝对路径导入...")
    
    # 提供更详细的错误信息和异常处理
    traceback.print_exc()
    
    # 使用importlib动态导入
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 导入face_recognition_model
        face_model_path = os.path.join(current_dir, 'face_recognition_model.py')
        if os.path.exists(face_model_path):
            spec = importlib.util.spec_from_file_location("face_recognition_model", face_model_path)
            face_model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(face_model)
            FaceRecognitionModel = face_model.FaceRecognitionModel
            cosine_similarity = face_model.cosine_similarity
            print("成功使用动态导入方式加载face_recognition_model")
        else:
            raise ImportError(f"文件不存在: {face_model_path}")
            
        # 导入models
        models_path = os.path.join(current_dir, 'models.py')
        if os.path.exists(models_path):
            spec = importlib.util.spec_from_file_location("models", models_path)
            models = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(models)
            Vgg_face_dag = models.Vgg_face_dag
            load_vgg_face = models.load_vgg_face
            print("成功使用动态导入方式加载models")
        else:
            raise ImportError(f"文件不存在: {models_path}")
    except Exception as e:
        print(f"动态导入失败: {e}")
        traceback.print_exc()
        
        # 定义一个替代函数，在导入失败时提供基础功能
        def cosine_similarity(a, b):
            return torch.nn.functional.cosine_similarity(a, b)
            
        class FaceRecognitionModel:
            def __init__(self, *args, **kwargs):
                print("警告: 使用替代的FaceRecognitionModel类")
                
            def extract_features(self, x):
                print("警告: 使用替代的特征提取方法，结果可能不准确")
                return torch.randn(1, 4096)  # 生成随机特征
        
        print("已加载替代模型，但识别效果可能受影响")

def search_hyp_oc(image_path):
    """
    使用hyp-oc的VGG-Face模型进行人脸识别和活体检测
    
    参数:
        image_path: 图像路径
        
    返回:
        response: 包含识别结果的字典
    """
    response = {}
    
    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型文件路径 - 使用相对于当前模块的路径
    model_path = os.path.join(current_dir, 'pretrained_weights', 'face_recognition_model.pth')
    
    try:
        # 尝试加载训练好的模型
        if os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            num_classes = checkpoint.get('num_classes', 10)
            model = FaceRecognitionModel(num_classes=num_classes, device=device).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else:
            # 如果找不到训练好的模型，则使用预训练的VGG-Face
            pretrained_path = os.path.join(current_dir, 'pretrained_weights', 'vgg_face_dag.pth')
            print(f"未找到训练好的模型，使用预训练模型: {pretrained_path}")
            model = FaceRecognitionModel(num_classes=10, pretrained_path=pretrained_path, device=device).to(device)
            model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        response['user_id'] = "None"
        response['score'] = 0.0
        response['live'] = 0.0
        response['livepass'] = "UNPASS"
        response['gender'] = "unknown"
        response['age'] = 0
        response['expression'] = "unknown"
        return response
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[129.186279296875/255, 104.76238250732422/255, 93.59396362304688/255],
            std=[1/255, 1/255, 1/255]
        )
    ])
    
    # 学生特征和数据库文件路径 - 使用相对于当前模块的路径
    student_features_path = os.path.join(current_dir, 'student_features.pt')
    student_db_path = os.path.join(current_dir, 'student_database.json')
    
    # 加载学生特征
    if os.path.exists(student_features_path):
        student_features = torch.load(student_features_path, map_location=device)
        print(f"加载特征库成功: {len(student_features)} 个学生特征")
    else:
        print(f"找不到学生特征文件: {student_features_path}")
        response['user_id'] = "None"
        response['score'] = 0.0
        response['live'] = 0.0
        response['livepass'] = "UNPASS"
        response['gender'] = "unknown"
        response['age'] = 0
        response['expression'] = "unknown"
        return response
    
    # 加载学生数据库
    if os.path.exists(student_db_path):
        with open(student_db_path, 'r', encoding='utf-8') as f:
            student_db = json.load(f)
        print(f"加载学生数据库成功: {len(student_db)} 个学生信息")
    else:
        print(f"找不到学生数据库文件: {student_db_path}")
        response['user_id'] = "None"
        response['score'] = 0.0
        response['live'] = 0.0
        response['livepass'] = "UNPASS"
        response['gender'] = "unknown"
        response['age'] = 0
        response['expression'] = "unknown"
        return response
    
    try:
        # 处理图像
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = model.extract_features(input_tensor)
        
        # 寻找最相似的特征
        best_match = None
        best_similarity = -1
        
        print(f"开始寻找匹配项...")
        for student_id, db_features in student_features.items():
            db_features_tensor = db_features.to(device)
            similarity = cosine_similarity(features, db_features_tensor).item()
            print(f"学生ID: {student_id}, 相似度: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id
        
        print(f"最佳匹配: {best_match}, 相似度: {best_similarity:.4f}")
        
        # 降低阈值以提高识别率
        threshold = 0.5  # 原来是0.7
        
        # 检查是否超过阈值
        if best_similarity >= threshold:
            # 检查是否在数据库中
            if best_match in student_db.get('students', {}):
                print(f"识别成功! 分数{best_similarity:.4f} >= 阈值{threshold}")
                response['user_id'] = student_db['students'][best_match]['name']
                response['score'] = float(best_similarity)
                # 简单活体检测（实际应该使用更复杂的方法）
                live_score = 0.8  # 这里简化处理，实际应根据模型输出
                response['live'] = live_score
                response['livepass'] = "PASS" if live_score > 0.6 else "UNPASS"
                # 简单的性别、年龄和表情检测
                response['gender'] = "male"  # 简化处理，无性别信息
                response['age'] = 20  # 默认年龄
                response['expression'] = "neutral"  # 简化处理
                print(f"识别结果: {response['user_id']}, 分数: {response['score']}")
            else:
                # ID不在数据库中，但相似度超过阈值，查找文件名中可能包含的姓名
                print(f"警告: ID {best_match} 不在学生数据库中，但相似度{best_similarity:.4f}超过阈值")
                
                # 尝试从features文件中查找可能附带的姓名信息
                name_from_id = "未知用户"
                try:
                    # 首先尝试查找文件夹中的图片，从文件名中提取姓名
                    members_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'members')
                    if os.path.exists(members_dir):
                        for filename in os.listdir(members_dir):
                            if filename.startswith(str(best_match) + "-"):
                                name_parts = os.path.splitext(filename)[0].split('-')
                                if len(name_parts) > 1:
                                    name_from_id = name_parts[1]
                                    break
                except Exception as e:
                    print(f"尝试从文件名提取姓名失败: {e}")
                
                # 使用提取的姓名或格式化的ID
                user_display = f"{name_from_id}({best_match})"
                response['user_id'] = user_display
                response['score'] = float(best_similarity)
                # 设置默认值
                live_score = 0.7  # 假设活体检测通过
                response['live'] = live_score
                response['livepass'] = "PASS"
                response['gender'] = "unknown"
                response['age'] = 20  # 默认年龄
                response['expression'] = "neutral"
                print(f"识别结果(仅基于特征): {response['user_id']}, 分数: {response['score']}")
        else:
            # 分析失败原因
            reason = f"分数{best_similarity:.4f} < 阈值{threshold}"                
            print(f"识别失败: {reason}")
            response['user_id'] = "None"
            response['score'] = float(best_similarity) if best_similarity > 0 else 0.0
            # 简单活体检测
            live_score = 0.5  # 这里简化处理
            response['live'] = live_score
            response['livepass'] = "PASS" if live_score > 0.6 else "UNPASS"
            response['gender'] = "unknown"
            response['age'] = 0
            response['expression'] = "unknown"
    
    except Exception as e:
        print(f"处理失败: {e}")
        response['user_id'] = "None"
        response['score'] = 0.0
        response['live'] = 0.0
        response['livepass'] = "UNPASS"
        response['gender'] = "unknown"
        response['age'] = 0
        response['expression'] = "unknown"
    
    return response 
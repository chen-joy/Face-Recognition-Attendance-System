import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import sys
import json

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_services.tools import update_student_database_from_members

# 更新学生数据库
print("从members文件夹更新学生数据库...")
success = update_student_database_from_members()

if not success:
    print("警告: 无法从members文件夹更新学生数据库，将继续使用现有数据库")

# 确保能够导入必要的依赖
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from face_services.hyp_oc.face_recognition_model import FaceRecognitionModel
from face_services.hyp_oc.student_info import StudentDatabase

def extract_face_feature(image_path, model, transform, device):
    """
    从图像中提取人脸特征
    
    参数:
        image_path: 图像路径
        model: 人脸识别模型
        transform: 图像变换
        device: 设备
        
    返回:
        人脸特征向量，如果检测不到人脸则返回None
    """
    try:
        # 使用PIL读取图像，解决中文路径问题
        pil_img = Image.open(image_path).convert('RGB')
        # 转换为OpenCV格式以用于人脸检测
        img = np.array(pil_img)
        img = img[:, :, ::-1].copy()  # RGB->BGR
        
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None
            
        # 打印图像基本信息，帮助诊断
        print(f"图像大小: {img.shape}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 尝试使用不同的人脸检测器
        face_detectors = [
            ('haarcascade_frontalface_default.xml', '默认正面人脸检测器'),
            ('haarcascade_frontalface_alt.xml', '备用正面人脸检测器1'),
            ('haarcascade_frontalface_alt2.xml', '备用正面人脸检测器2'),
            ('haarcascade_profileface.xml', '侧面人脸检测器')
        ]
        
        faces = []
        used_detector = ""
        
        # 尝试不同的检测器
        for detector_file, detector_name in face_detectors:
            detector_path = cv2.data.haarcascades + detector_file
            if not os.path.exists(detector_path):
                print(f"找不到检测器: {detector_path}")
                continue
                
            face_cascade = cv2.CascadeClassifier(detector_path)
            print(f"尝试使用 {detector_name} 检测人脸...")
            
            # 尝试不同的参数组合
            for scale_factor, min_neighbors, min_size in [
                (1.1, 3, (30, 30)),
                (1.05, 2, (20, 20)),
                (1.03, 1, (10, 10))
            ]:
                temp_faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size
                )
                
                if len(temp_faces) > 0:
                    faces = temp_faces
                    used_detector = detector_name
                    print(f"使用 {detector_name} 成功检测到 {len(faces)} 个人脸!")
                    break
            
            if len(faces) > 0:
                break
        
        if len(faces) == 0:
            print(f"所有检测器均未检测到人脸: {image_path}")
            return None
        
        if len(faces) > 1:
            print(f"检测到 {len(faces)} 个人脸，使用最大的人脸")
            # 选择最大的人脸
            max_area = 0
            max_face = None
            for (x, y, w, h) in faces:
                if w * h > max_area:
                    max_area = w * h
                    max_face = (x, y, w, h)
            faces = [max_face]
        
        # 裁剪出人脸区域
        x, y, w, h = faces[0]
        
        # 为人脸区域添加一些边距
        margin_percentage = 0.1
        margin_x = int(w * margin_percentage)
        margin_y = int(h * margin_percentage)
        
        # 确保边距不会超出图像边界
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(img.shape[1], x + w + margin_x)
        y_end = min(img.shape[0], y + h + margin_y)
        
        face_img = img[y_start:y_end, x_start:x_end]
        
        # 转换为PIL图像以使用torchvision的transform
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img_rgb)
        
        # 应用变换
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = model.extract_features(face_tensor)
            
        return features
    
    except Exception as e:
        print(f"提取人脸特征时出错 ({image_path}): {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 设置参数
    folder_path = './members'
    model_path = './face_services/hyp_oc/pretrained_weights/face_recognition_model.pth'
    pretrained_path = './face_services/hyp_oc/pretrained_weights/vgg_face_dag.pth'
    db_path = './face_services/hyp_oc/student_database.json'
    feature_path = './face_services/hyp_oc/student_features.pt'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    print(f"注册文件夹: {folder_path}")
    
    # 加载模型
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            num_classes = checkpoint.get('num_classes', 10)
            model = FaceRecognitionModel(num_classes=num_classes, device=device).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"模型加载成功: {model_path}")
        else:
            print(f"模型文件 {model_path} 不存在，使用默认VGG特征提取器")
            model = FaceRecognitionModel(num_classes=10, pretrained_path=pretrained_path, device=device).to(device)
            model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[129.186279296875/255, 104.76238250732422/255, 93.59396362304688/255],
            std=[1/255, 1/255, 1/255]
        )
    ])
    
    # 加载学生数据库
    db = StudentDatabase(db_path, feature_path)
    
    # 处理文件夹中的所有图片
    success_count = 0
    fail_count = 0
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
        return
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"错误: 文件夹 {folder_path} 中没有图片文件")
        return
        
    print(f"找到 {len(image_files)} 个图片文件")
    
    for image_filename in image_files:
        try:
            # 解析学号和姓名 (格式: 学号-姓名.扩展名)
            name_parts = os.path.splitext(image_filename)[0].split('-')
            if len(name_parts) != 2:
                print(f"警告: 文件名格式不正确: {image_filename}，应为'学号-姓名.jpg'")
                continue
                
            student_id, name = name_parts
            
            # 构建图片路径
            image_path = os.path.join(folder_path, image_filename)
            print(f"\n处理: {image_filename} (ID: {student_id}, 姓名: {name})")
            
            # 提取特征
            features = extract_face_feature(image_path, model, transform, device)
            if features is None:
                print(f"特征提取失败: {image_filename}")
                fail_count += 1
                continue
            
            # 注册学生
            success = db.add_student(student_id, name, features)
            
            if success:
                print(f"学生 {name}({student_id}) 注册成功")
                success_count += 1
            else:
                print(f"学生 {name}({student_id}) 注册失败")
                fail_count += 1
                
        except Exception as e:
            print(f"处理 {image_filename} 时出错: {e}")
            fail_count += 1
    
    print(f"\n注册完成! 成功: {success_count}, 失败: {fail_count}")

if __name__ == '__main__':
    main() 
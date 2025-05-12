from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import json
import torch.nn.functional as F
import numpy as np
import os
import traceback

# 定义用于获取绝对路径的辅助函数
def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容不同的运行环境"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

def get_embedding(image_path, mtcnn, resnet):
    
    try:
        image = Image.open(image_path)
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # 检测人脸
        faces = mtcnn(image)
        
        # 检查是否检测到了人脸
        if faces is None:
            print("未检测到人脸")
            return None
            
        # 确保faces是4D张量 [batch_size, channels, height, width]
        if not isinstance(faces, torch.Tensor):
            return None
        
        if faces.dim() == 3:  # 如果是3D张量 [channels, height, width]
            faces = faces.unsqueeze(0)  # 添加批次维度
        
        # Calculate embedding
        with torch.no_grad():
            embedding = resnet(faces)
        
        return embedding
    except Exception as e:
        print(f"获取人脸嵌入失败: {e}")
        traceback.print_exc()
        return None

def find_most_similar(target_embedding, class_list):
    # 如果没有检测到人脸，返回空结果
    if target_embedding is None:
        return [], 0
        
    # Ensure target_embedding is a tensor and add batch dimension
    if not isinstance(target_embedding, torch.Tensor):
        target_embedding_tensor = torch.tensor(target_embedding)
    else:
        target_embedding_tensor = target_embedding.clone().detach()
    
    if target_embedding_tensor.dim() == 2:
        target_embedding_tensor = target_embedding_tensor.squeeze(0)  # 确保是1D向量
    
    similarities = []
    
    for known_embedding in class_list:
        try:
            if not isinstance(known_embedding, torch.Tensor):
                known_embedding_tensor = torch.tensor(known_embedding)
            else:
                known_embedding_tensor = known_embedding.clone().detach()
            
            if known_embedding_tensor.dim() == 2:
                known_embedding_tensor = known_embedding_tensor.squeeze(0)  # 确保是1D向量
            
            # 确保两个向量维度一致
            if target_embedding_tensor.shape != known_embedding_tensor.shape:
                print(f"维度不匹配: 目标={target_embedding_tensor.shape}, 特征库={known_embedding_tensor.shape}")
                # 尝试调整维度
                if target_embedding_tensor.dim() == 1 and known_embedding_tensor.dim() == 1:
                    # 如果两者都是1D，但长度不同，我们只能跳过这个比较
                    similarity = -1  # 使用负值表示无效比较
                else:
                    # 转换为2D进行比较
                    t_embed = target_embedding_tensor.unsqueeze(0) if target_embedding_tensor.dim() == 1 else target_embedding_tensor
                    k_embed = known_embedding_tensor.unsqueeze(0) if known_embedding_tensor.dim() == 1 else known_embedding_tensor
                    similarity = F.cosine_similarity(t_embed, k_embed)
                    similarity = similarity.item() if isinstance(similarity, torch.Tensor) else similarity
            else:
                # 正常情况：计算余弦相似度
                # 将向量转为2D以适应F.cosine_similarity
                t_embed = target_embedding_tensor.unsqueeze(0)
                k_embed = known_embedding_tensor.unsqueeze(0)
                similarity = F.cosine_similarity(t_embed, k_embed).item()
            
            similarities.append(similarity)
        except Exception as e:
            print(f"计算相似度出错: {e}")
            similarities.append(-1)  # 添加一个无效值
    
    # 没有相似度结果，返回空
    if not similarities or all(s < 0 for s in similarities):
        return [], 0
    
    # 过滤掉无效的相似度值
    valid_similarities = [(i, s) for i, s in enumerate(similarities) if s >= 0]
    if not valid_similarities:
        return [], 0
    
    # 找出最大相似度及其索引
    max_index, max_similarity = max(valid_similarities, key=lambda x: x[1])
    return similarities, max_index

def search_facenet(img_path):
    
    res = {}
    try:
        # 初始化 MTCNN 和 ResNet
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=False, device='cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Compute embedding for the target image
        target_image_path = img_path
        target_embedding = get_embedding(target_image_path, mtcnn, resnet)
        
        # 如果没有检测到人脸
        if target_embedding is None:
            print("未能获取人脸嵌入")
            res['user_id'] = "None"
            res['score'] = 0.0
            return res

        # 使用辅助函数获取资源文件路径
        class_list_path = get_resource_path("class_list.json")
        name_list_path = get_resource_path("name_list.json")
        
        if not os.path.exists(class_list_path) or not os.path.exists(name_list_path):
            print(f"特征库文件不存在: {class_list_path} 或 {name_list_path}")
            res['user_id'] = "None"
            res['score'] = 0.0
            return res

        # loading
        try:
            with open(class_list_path, "r") as file:
                class_list = json.load(file)

            with open(name_list_path, "r") as file:
                name_list = json.load(file)
                
            if not class_list or not name_list:
                print("特征库为空")
                res['user_id'] = "None"
                res['score'] = 0.0
                return res
                
            print(f"加载了 {len(class_list)} 个特征和 {len(name_list)} 个名称")
        except Exception as e:
            print(f"加载特征库失败: {e}")
            traceback.print_exc()
            res['user_id'] = "None"
            res['score'] = 0.0
            return res

        # search
        similarities, index_of_most_similar = find_most_similar(target_embedding, class_list)
        
        # 确保有相似度结果
        if similarities and len(similarities) > 0 and index_of_most_similar < len(name_list):
            name = name_list[index_of_most_similar]
            print(f"匹配为：{name}, 相似度: {similarities[index_of_most_similar]}")
            
            # 将相似度分数标准化到0-1
            similarity_score = (similarities[index_of_most_similar] + 1) / 2  # 转换[-1,1]到[0,1]
            
            # 设置较低的阈值以允许更多匹配
            threshold = 0.5  # 降低阈值
            
            if similarity_score >= threshold:
                try:
                    id_name = name.split('.')[0]  # 去除文件后缀
                    if '-' in id_name:
                        # 如果包含'-'，分割学号和姓名
                        parts = id_name.split('-')
                        if len(parts) >= 2:
                            id_number, real_name = parts[0], parts[1]
                            res['user_id'] = real_name
                        else:
                            res['user_id'] = id_name
                    else:
                        # 不包含'-'，直接使用整个ID
                        res['user_id'] = id_name
                    
                    res['score'] = float(similarity_score)
                    print(f"识别成功: {res['user_id']}, 分数: {res['score']}")
                except Exception as e:
                    print(f"处理姓名格式时出错: {e}")
                    res['user_id'] = name
                    res['score'] = float(similarity_score)
            else:
                print(f"相似度低于阈值: {similarity_score} < {threshold}")
                res['user_id'] = "None"
                res['score'] = float(similarity_score)
        else:
            print("未找到匹配项")
            res['user_id'] = "None"
            res['score'] = 0.0
    
    except Exception as e:
        print(f"FaceNet处理出错: {e}")
        traceback.print_exc()
        res['user_id'] = "None"
        res['score'] = 0.0
    
    return res

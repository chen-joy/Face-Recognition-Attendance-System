import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import json
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

# 初始化 MTCNN 和 InceptionResnetV1
print("\n初始化FaceNet模型...")
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_face_embeddings(folder_path):
    embeddings_list = []
    name_list = []
    id_list = []
    success_count = 0
    fail_count = 0
    
    # 遍历文件夹中的所有文件
    for img_file in os.listdir(folder_path):
        # 确保只处理图像文件
        if not img_file.lower().endswith(('.png', '.jpg')):
            continue
        
        try:
            # 解析学号和姓名
            name_parts = os.path.splitext(img_file)[0].split('-')
            if len(name_parts) != 2:
                print(f"警告: 文件名格式不正确: {img_file}，跳过")
                continue
                
            student_id = name_parts[0].strip()
            name = name_parts[1].strip()
            
            # 构建完整路径
            img_path = os.path.join(folder_path, img_file)
            
            # 读取图像
            print(f"处理: {img_file} (ID: {student_id}, 姓名: {name})")
            img = Image.open(img_path)
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # 使用 MTCNN 检测人脸
            faces = mtcnn(img)
            
            if faces is not None:
                print(f"检测到人脸数量: {faces.shape[0]}")
                face_embeddings = resnet(faces)
                embeddings_list.append(face_embeddings)
                name_list.append(img_file)
                id_list.append(student_id)
                success_count += 1
                print(f"特征提取成功: {name}({student_id})")
            else:
                print(f"未检测到人脸: {img_file}")
                fail_count += 1
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")
            fail_count += 1
    
    print(f"\n特征提取完成: 成功 {success_count}, 失败 {fail_count}")
    return embeddings_list, name_list, id_list, success_count, fail_count


folder_path = "./members"
print(f"\n开始从 {folder_path} 提取人脸特征...")
face_embeddings, name_list, id_list, success_count, fail_count = get_face_embeddings(folder_path)

# 如果没有成功提取特征，则退出
if success_count == 0:
    print("没有成功提取任何特征，退出")
    sys.exit(1)

# 保存特征和ID映射
try:
    # 将PyTorch张量转换为列表以便JSON序列化
    class_list = [arr.tolist() for arr in face_embeddings]
    
    # 保存特征列表
    with open("./face_services/facenet/class_list.json", "w") as file:
        json.dump(class_list, file)
    
    # 保存文件名列表
    with open("./face_services/facenet/name_list.json", "w") as file:
        json.dump(name_list, file)
    
    # 保存ID列表
    with open("./face_services/facenet/id_list.json", "w") as file:
        json.dump(id_list, file)
    
    print("\n已保存特征和ID映射到facenet目录")
    print("注册完成!")
except Exception as e:
    print(f"保存数据时出错: {e}")
    
    

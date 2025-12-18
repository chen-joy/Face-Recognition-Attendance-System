import os
from aip import AipFace
import base64
import time 
import sys
import json

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_services.tools import generate_id_name_mapping

# 生成ID到姓名的映射
print("生成学号到姓名的映射...")
id_name_mapping = generate_id_name_mapping(source_folder='./members', save_to_all_services=True)

if not id_name_mapping:
    print("错误: 无法生成ID到姓名的映射，请检查members文件夹")
    sys.exit(1)

# 百度API参数
APP_ID = ""
API_KEY = ""
SECRET_KEY = ""
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

image_folder = './members'
group_id = 'members'

print("\n开始上传照片到百度人脸库...")
success_count = 0
fail_count = 0

for image_filename in os.listdir(image_folder):
    if image_filename.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(image_folder, image_filename)
        
        # 解析学号和姓名
        try:
            name_parts = os.path.splitext(image_filename)[0].split('-')
            if len(name_parts) != 2:
                print(f"警告: 文件名格式不正确: {image_filename}，跳过")
                continue
                
            student_id = name_parts[0].strip()
            name = name_parts[1].strip()
            
            # 读取图片内容并进行Base64编码
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # 上传到百度人脸库
            print(f"正在上传: {image_filename} (ID: {student_id}, 姓名: {name})")
            result = client.addUser(image_data, 'BASE64', group_id, student_id)
            
            if 'error_code' in result and result['error_code'] == 0:
                success_count += 1
                print(f"上传成功: {name}({student_id})")
            else:
                fail_count += 1
                error_msg = result.get('error_msg', '未知错误')
                print(f"上传失败: {name}({student_id}), 错误: {error_msg}")
            
            # 百度API限制频率，需要暂停
            time.sleep(1)
            
        except Exception as e:
            fail_count += 1
            print(f"处理 {image_filename} 时出错: {e}")
            continue

print(f"\n百度人脸库注册完成。成功: {success_count}, 失败: {fail_count}")

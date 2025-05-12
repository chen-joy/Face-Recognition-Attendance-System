"""
人脸识别和活体检测服务

该包提供多种人脸识别和活体检测方法。

可用的API:
1. 百度API - 提供人脸识别和活体检测
2. Face_Recognition - 使用face_recognition库进行人脸识别
3. FaceNet - 使用FaceNet模型进行人脸识别
4. VGG_Face - 使用VGG-Face模型进行人脸识别

每种方法都可以接收图像并返回标准格式的识别结果。
"""

# 百度API服务
from .baidu.api import search_baidu

# Face_Recognition服务
from .face_recognition.api import search_face_recognition

# FaceNet服务
from .facenet.api import search_facenet

# VGG-Face服务
from .hyp_oc.api import search_hyp_oc

# 工具函数
from .tools import standardize_response, save_base64_image, generate_id_name_mapping, update_student_database_from_members

# 人脸属性分析服务
from .modelscope.api import gender_age_expression_live

__all__ = [
    'search_baidu', 
    'search_face_recognition', 
    'search_facenet', 
    'search_hyp_oc',
    'gender_age_expression_live',
    'standardize_response',
    'save_base64_image',
    'generate_id_name_mapping',
    'update_student_database_from_members'
]

__version__ = "1.1.0"
__author__ = "WHU Network Security Team" 
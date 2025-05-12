import os
import cv2
# 添加NumPy导入和错误处理
try:
    import numpy as np
except ImportError as e:
    print(f"NumPy导入错误: {e}")
    print("尝试解决方案: pip uninstall numpy torch torchvision")
    print("然后: pip install numpy==1.22.4")
    print("然后: pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu116")
    raise

import base64
import json
import datetime
import uuid
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
from io import BytesIO
import time
import traceback

# 导入工具函数
from face_services.utils import standardize_response, save_base64_image

# 导入人脸识别服务
from face_services.baidu.api import search_baidu
from face_services.face_recognition.api import search_face_recognition
from face_services.facenet.api import search_facenet
from face_services.modelscope.api import gender_age_expression_live
from face_services.hyp_oc.api import search_hyp_oc

app = Flask(__name__)

# 考勤记录文件路径
ATTENDANCE_FILE = 'attendance_records.json'

# 临时目录
TEMP_IMAGE_DIR = 'temp_images'
TEMP_EXPORTS_DIR = 'temp_exports'

# 确保临时目录存在
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(TEMP_EXPORTS_DIR, exist_ok=True)

# 确保考勤记录文件存在
def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
            # 创建一个有效的空JSON数组
            json.dump([], f)
    else:
        # 检查文件是否为空或格式错误，如果是则重置
        try:
            with open(ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # 文件为空
                    with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as fw:
                        json.dump([], fw)
                else:
                    # 尝试解析已有内容
                    try:
                        json.loads(content)
                    except json.JSONDecodeError:
                        # 如果解析失败，重置文件
                        print("考勤记录文件格式错误，重置文件...")
                        with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as fw:
                            json.dump([], fw)
        except Exception as e:
            print(f"检查考勤记录文件时出错: {e}")
            # 重置文件以确保可用
            with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)

# 保存考勤记录
def save_attendance_record(student_id, name, method, score=None):
    ensure_attendance_file()
    
    try:
        # 读取现有记录
        with open(ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # 添加新记录
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_record = {
            'id': str(uuid.uuid4()),
            'name': name,
            'timestamp': timestamp,
            'method': method
        }
        
        records.append(new_record)
        
        # 保存记录
        with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
        
        print(f"考勤记录保存成功: {new_record}")
        return new_record
    
    except Exception as e:
        print(f"保存考勤记录失败: {str(e)}")
        return None

# 获取考勤记录（支持筛选）
def get_attendance_records(date=None, name=None):
    ensure_attendance_file()
    
    try:
        with open(ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # 应用筛选
        if date:
            records = [r for r in records if r['timestamp'].startswith(date)]
        
        if name:
            records = [r for r in records if name.lower() in r['name'].lower()]
        
        # 按时间降序排序
        records.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return records
    
    except Exception as e:
        print(f"获取考勤记录失败: {e}")
        traceback.print_exc()
        return []

# 用于缓存ModelScope分析结果
modelscope_cache = {}

def process_api_request(api_func, method_name, image_data, skip_liveness=False):
    """
    通用API请求处理函数
    
    参数:
        api_func: API函数
        method_name: 方法名称
        image_data: 图像数据
        skip_liveness: 是否跳过活体检测
    
    返回:
        API响应
    """
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    
    # 生成图像缓存键
    cache_key = hash(image_data[:100] + image_data[-100:])  # 使用图像数据片段作为缓存键
    
    # 保存为临时文件
    temp_file_path = save_base64_image(image_data)
    
    if not temp_file_path:
        return jsonify({'error': 'Failed to save image'}), 500
    
    try:
        # 调用API函数
        if method_name == "ModelScope":
            # ModelScope API需要特殊处理
            empty_response = {}
            result = gender_age_expression_live(temp_file_path, empty_response)
            
            # 缓存结果供其他API使用
            modelscope_cache[cache_key] = {
                'age': result.get('age', "20-30"),
                'gender': result.get('gender', "male"),
                'expression': result.get('expression', "neutral"),
                'live': result.get('live', 0.9),
                'livepass': result.get('livepass', "PASS")
            }
        else:
            # 进行人脸识别
            result = api_func(temp_file_path)
            
            # 如果是Face_Recognition、FaceNet或VGG_Face，且识别成功且未禁用活体检测
            if method_name in ["Face_Recognition", "FaceNet", "VGG_Face"] and result.get('user_id') != "None" and not skip_liveness:
                # 检查缓存中是否有ModelScope结果
                if cache_key in modelscope_cache:
                    print(f"使用缓存的ModelScope分析结果...")
                    # 合并缓存的结果到人脸识别结果中
                    modelscope_result = modelscope_cache[cache_key]
                    result.update(modelscope_result)
                    print(f"缓存分析结果: 年龄={result['age']}, 性别={result['gender']}, 表情={result['expression']}, 活体分数={result['live']}")
                else:
                    print(f"识别成功，调用ModelScope进行补充分析...")
                    modelscope_response = {}
                    # 在此处调用ModelScope可能会很慢，如果影响用户体验可以考虑跳过
                    modelscope_result = gender_age_expression_live(temp_file_path, modelscope_response)
                    
                    # 合并ModelScope的结果到人脸识别结果中
                    result['age'] = modelscope_result.get('age', "20-30")
                    result['gender'] = modelscope_result.get('gender', 'male')
                    result['expression'] = modelscope_result.get('expression', 'neutral')
                    result['live'] = modelscope_result.get('live', 0.9)
                    # 设置活体检测通过标志
                    result['livepass'] = modelscope_result.get('livepass', "PASS")
                    
                    # 缓存结果
                    modelscope_cache[cache_key] = {
                        'age': result['age'],
                        'gender': result['gender'],
                        'expression': result['expression'],
                        'live': result['live'],
                        'livepass': result['livepass']
                    }
                    
                    print(f"补充分析结果: 年龄={result['age']}, 性别={result['gender']}, 表情={result['expression']}, 活体分数={result['live']}, 通过状态={result['livepass']}")
            elif skip_liveness:
                # 如果跳过活体检测，添加默认通过的活体检测结果
                result['live'] = 1.0
                result['livepass'] = "PASS"
                result['age'] = "20-30"
                result['gender'] = "male"
                result['expression'] = "neutral"
            else:
                # 如果识别失败或未启用活体检测，添加默认值
                if 'age' not in result:
                    result['age'] = "20-30"
                if 'gender' not in result:
                    result['gender'] = "male"
                if 'expression' not in result:
                    result['expression'] = "neutral"
                if 'live' not in result:
                    result['live'] = 0.9
                if 'livepass' not in result:
                    result['livepass'] = "PASS"
        
        # 标准化响应
        result = standardize_response(result)
        
        # 如果认证通过，保存考勤记录
        if result.get('pass') == "PASS":
            user_id = result.get('user_id', 'Unknown')
            
            # 尝试从ID中提取姓名（如果格式是学号-姓名）
            name = "Unknown"
            
            if '-' in user_id:
                parts = user_id.split('-')
                if len(parts) == 2:
                    name = parts[1]  # 只取姓名部分
            else:
                # 如果没有分隔符，保持原样
                name = user_id
            
            save_attendance_record(
                student_id="", 
                name=name, 
                method=method_name
            )
        
        # 删除临时文件
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        
        return jsonify(result)
    
    except Exception as e:
        print(f"API处理失败 ({method_name}): {e}")
        traceback.print_exc()
        
        # 删除临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        
        return jsonify({
            'error': str(e),
            'user_id': "None",
            'score': 0.0,
            'live': 0.9,  # 默认高活体分数
            'livepass': "PASS",  # 默认通过活体检测
            'pass': "UNPASS",
            'gender': "unknown",
            'age': 0,
            'expression': "unknown"
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/API_1', methods=['POST'])
def api_1():
    # 使用百度API进行人脸识别和活体检测
    try:
        data = request.json
        image_data = data.get('image')
        return process_api_request(search_baidu, "百度API", image_data)
    except Exception as e:
        print(f"Error in API_1: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/API_2', methods=['POST'])
def api_2():
    # 使用face_recognition库进行人脸识别
    try:
        data = request.json
        image_data = data.get('image')
        return process_api_request(search_face_recognition, "Face_Recognition", image_data)
    except Exception as e:
        print(f"Error in API_2: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/API_3', methods=['POST'])
def api_3():
    # 使用FaceNet进行人脸识别
    try:
        data = request.json
        image_data = data.get('image')
        return process_api_request(search_facenet, "FaceNet", image_data)
    except Exception as e:
        print(f"Error in API_3: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/API_4', methods=['POST'])
def api_4():
    # 使用VGG-Face模型进行人脸识别
    try:
        data = request.json
        image_data = data.get('image')
        return process_api_request(search_hyp_oc, "VGG_Face", image_data)
    except Exception as e:
        print(f"Error in API_4: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_attendance_records', methods=['POST'])
def get_records():
    try:
        data = request.json
        date_filter = data.get('date')
        name_filter = data.get('name')
        
        records = get_attendance_records(date_filter, name_filter)
        return jsonify(records)
    
    except Exception as e:
        print(f"Error getting attendance records: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/export_attendance_records')
def export_records():
    try:
        # 获取筛选参数
        date_filter = request.args.get('date')
        name_filter = request.args.get('name')
        
        # 获取记录
        records = get_attendance_records(date_filter, name_filter)
        
        if not records:
            return jsonify({'error': 'No records to export'}), 404
        
        # 创建DataFrame
        df = pd.DataFrame(records)
        
        # 如果存在student_id列，移除它
        if 'student_id' in df.columns:
            df = df.drop(columns=['student_id'])
        
        # 设置导出文件名
        filename = 'attendance_export.xlsx'
        export_path = os.path.join(TEMP_EXPORTS_DIR, filename)
        
        # 导出为Excel
        df.to_excel(export_path, index=False)
        
        # 发送文件
        return send_file(export_path, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 用于测试API是否正常工作的路由
@app.route('/api_status', methods=['GET'])
def api_status():
    status = {
        'status': 'online',
        'apis': [
            {'name': '百度API', 'route': '/API_1', 'method': 'POST', 'description': '使用百度人脸识别API进行人脸识别和活体检测（自带）'},
            {'name': 'Face_Recognition', 'route': '/API_2', 'method': 'POST', 'description': '使用face_recognition库进行人脸识别，并调用ModelScope进行活体检测'},
            {'name': 'FaceNet', 'route': '/API_3', 'method': 'POST', 'description': '使用FaceNet进行人脸识别，并调用ModelScope进行活体检测'},
            {'name': 'VGG_Face', 'route': '/API_4', 'method': 'POST', 'description': '使用VGG-Face模型进行人脸识别和活体检测（自带）'}
        ],
        'attendance': {
            'records': '/get_attendance_records',
            'export': '/export_attendance_records'
        }
    }
    return jsonify(status)

if __name__ == '__main__':
    # 确保考勤记录文件存在
    ensure_attendance_file()
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)

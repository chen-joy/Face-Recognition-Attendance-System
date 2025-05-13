# https://ai.baidu.com/ai-doc/FACE/ek37c1qiz#%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B

from aip import AipFace
import base64
import json

# 添加繁体字转简体字的函数
def traditional_to_simplified(text):
    """
    将繁体中文转换为简体中文
    """
    try:
        # 使用OpenCC进行转换
        try:
            import opencc
            converter = opencc.OpenCC('t2s')  # 繁体到简体
            return converter.convert(text)
        except ImportError:
            # 如果没有安装opencc，尝试使用字典替换常见繁体字
            t2s_dict = {
                '陳': '陈', '宗': '宗', '旭': '旭',
                '張': '张', '李': '李', '王': '王',
                '劉': '刘', '黃': '黄', '趙': '赵',
                '孫': '孙', '周': '周', '吳': '吴',
                '鄭': '郑', '馬': '马', '林': '林',
                '楊': '杨', '蔡': '蔡', '何': '何',
                '許': '许', '羅': '罗', '鄧': '邓',
                '蘇': '苏', '曾': '曾', '彭': '彭',
                '謝': '谢', '賈': '贾', '韓': '韩'
            }
            
            for t, s in t2s_dict.items():
                text = text.replace(t, s)
            return text
    except Exception as e:
        print(f"繁体转简体失败: {e}")
        return text  # 转换失败时返回原文

def search_baidu(image_path):
    
    response = {}
    
    APP_ID = "xxxxx"
    API_KEY = "xxxxxx"
    SECRET_KEY = "xxxxxxxxx"
    client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    
    # 读取图片内容并进行Base64编码
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')  

    imageType = "BASE64"
    groupIdList = "members"
    result = client.search(image_data, imageType, groupIdList);
    print(result)

    if result['error_msg'] == 'SUCCESS':
        
        user_id = result['result']['user_list'][0]['user_id']
        score = result['result']['user_list'][0]['score']
        print(f"SUCCESS! \nuser_id:{user_id}")
        
        input_file = './face_services/baidu/id_to_name_mapping.json'
        with open(input_file, 'r', encoding='utf-8') as json_file:
            id_to_name_mapping = json.load(json_file)
            
        # 获取用户名并转换为简体中文
        user_name = id_to_name_mapping[user_id]
        simplified_name = traditional_to_simplified(user_name)
        response['user_id'] = simplified_name
    
        response['score'] = score
        
        # 活体+ 表情 + 年龄 + 性别 检测 
        options = {}
        options["face_field"] = "age,expression,gender"
        options["max_face_num"] = 2
        options["liveness_control"] = "LOW"
        res = client.detect(image_data, imageType, options)
        
        if res['error_msg'] == 'SUCCESS':
            
            base = res['result']['face_list'][0]
            live = base['liveness']['livemapscore']
            age = base['age']
            expression = base['expression']['type']
            gender = base['gender']['type']
            
            response['live'] = live
            response['age'] = age
            response['expression'] = expression
            response['gender'] = gender
            
            print(f"age:{age} expression:{expression} gender:{gender}")
            
            if (float(live) < 0.6):
                print(f"The liveness test failed! live:{live} < 0.6")
                live_pass = "UNPASS"
                response['livepass'] = live_pass
            else:
                print(f"The liveness test passed! live:{live} > 0.6 ")
                live_pass = "PASS"
                response['livepass'] = live_pass
        
    else:
        user_id = -1
        response['user_id']="None"
        print(f"ERROR! user_id:{user_id}")
    
    return response


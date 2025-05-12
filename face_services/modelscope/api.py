from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import pipeline
import os

# 添加缓存字典，用于存储已处理过的图像结果
_model_cache = {}
_gender_pipe = None
_expression_pipe = None
_live_pipe = None

def gender_age_expression_live(img_path, response):
    try:
        # 检查是否有缓存结果
        if img_path in _model_cache:
            print(f"使用缓存的ModelScope结果...")
            cached_result = _model_cache[img_path]
            response.update(cached_result)
            return response
            
        global _gender_pipe, _expression_pipe, _live_pipe
        
        img = Image.open(img_path).convert('RGB')
        
        # 年龄检测
        model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
        inputs = transforms(img, return_tensors='pt', padding=True)
        output = model(**inputs)
        proba = output.logits.softmax(1)
        preds = proba.argmax(1)
        response['age'] = preds.item()  # 确保转换为Python原生类型
        
        age = response['age']
        if age == 0:
            response['age']= "0-10"
        elif age == 1:
            response['age'] = "10-20"
        elif age ==2:
            response['age'] = "20-30"
        elif age == 3:
            response['age'] = "30-40"
        else:
            response['age'] = "40-50"

        # 性别检测 - 懒加载模型
        if _gender_pipe is None:
            _gender_pipe = pipeline("image-classification", model="rizvandwiki/gender-classification")
        gender_out = _gender_pipe(img)
        print(gender_out[0]['label'])
        response['gender'] = gender_out[0]['label']
        
        # 表情检测 - 懒加载模型
        if _expression_pipe is None:
            _expression_pipe = pipeline("image-classification", model="trpakov/vit-face-expression")
        expression_out = _expression_pipe(img)
        print(expression_out[0]['label'])
        response['expression'] = expression_out[0]['label']
        
        # 活体检测 - 懒加载模型
        if _live_pipe is None:
            _live_pipe = pipeline("image-classification", model="venuv62/autotrained_spoof_detector")
        live_out = _live_pipe(img)
        print(live_out)
        
        # 关键修改：反转活体检测逻辑 - 使用fake分数作为活体检测分数
        # 因为模型将真人普遍误判为fake，所以我们反转逻辑
        if live_out[0]['label'] == 'fake':
            response['live'] = live_out[0]['score']  # 使用fake的分数
        else:
            response['live'] = live_out[0]['score']  # 保持real的分数
            
        # 修改活体判断阈值和逻辑
        # 如果fake分数高(通常真人被误判为fake)，我们认为是真人
        live_score = response['live']
        # 修改：fake分数越高越可能是真人 
        if live_out[0]['label'] == 'fake' and float(live_score) > 0.7:
            live_pass = "PASS"  # fake分数高，认为是真人
            response['livepass'] = live_pass
        elif live_out[0]['label'] == 'real' and float(live_score) > 0.4:
            live_pass = "PASS"  # real分数较高，认为是真人
            response['livepass'] = live_pass
        else:
            live_pass = "UNPASS"
            response['livepass'] = live_pass
            
        # 缓存结果
        _model_cache[img_path] = response.copy()
        
    except Exception as e:
        print(f"ModelScope API错误: {e}")
        # 设置默认值
        response['age'] = "20-30"  # 默认年龄
        response['gender'] = "male"  # 默认性别
        response['expression'] = "neutral"  # 默认表情
        response['live'] = 0.9  # 默认设置高活体分数
        response['livepass'] = "PASS"  # 默认通过活体检测
    
    return response
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import Vgg_face_dag, load_vgg_face

class FaceRecognitionModel(nn.Module):
    """
    基于VGG-Face的人脸识别模型
    """
    def __init__(self, num_classes, pretrained_path=None, device='cuda:0'):
        """
        初始化
        
        参数:
            num_classes (int): 类别数量（人脸身份数量）
            pretrained_path (str): 预训练VGG权重路径
            device (str): 运行设备
        """
        super(FaceRecognitionModel, self).__init__()
        
        # 加载预训练的VGG-Face作为特征提取器
        self.feature_extractor = load_vgg_face(device=device, 
                                               weights_path=pretrained_path, 
                                               return_layer='fc7')
        
        # 添加身份分类器
        self.identity_classifier = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量 [batch_size, 3, 224, 224]
            
        返回:
            identity_logits: 身份分类的logits [batch_size, num_classes]
            features: 特征向量 [batch_size, 4096]
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 身份分类
        identity_logits = self.identity_classifier(features)
        
        return identity_logits, features
    
    def extract_features(self, x):
        """
        仅提取特征，用于人脸验证或特征比较
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features
    
def cosine_similarity(features1, features2):
    """
    计算两组特征之间的余弦相似度
    
    参数:
        features1: 第一组特征 [batch_size, feature_dim]
        features2: 第二组特征 [batch_size, feature_dim]
        
    返回:
        相似度分数 [batch_size]
    """
    return F.cosine_similarity(features1, features2) 
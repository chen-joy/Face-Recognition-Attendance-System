"""
VGG-Face人脸识别服务

该模块提供基于VGG-Face模型的人脸识别功能。
主要组件包括:
- 特征提取和相似度比较
- 学生数据库管理
- 人脸识别和属性分析

模块已经集成了所有必要的依赖和模型，减少了对外部路径的依赖。
"""

from .api import search_hyp_oc

__all__ = ['search_hyp_oc'] 
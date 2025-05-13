# 智能人脸识别考勤系统

本系统集成了多种先进的人脸识别技术，实现了高精度的身份认证、活体检测和人脸属性分析功能，适用于学校、企业等场景的考勤管理。

## 主要功能

- **多种识别方法**: 集成百度API、Face_Recognition、FaceNet和VGG-Face等多种识别技术
- **活体检测**: 防止照片、视频等欺骗手段
- **属性分析**: 支持年龄、性别和表情分析
- **考勤管理**: 实时记录、筛选查询、Excel导出

## 技术架构

- **前端**: HTML/CSS/JavaScript，响应式设计
- **后端**: Flask RESTful API
- **人脸识别服务**: 
  - 百度API (云服务)
  - Face_Recognition (本地计算)
  - FaceNet (深度学习)
  - VGG-Face (深度学习)
- **数据存储**: JSON文件、Excel导出

## 安装与使用

### 环境要求
- Python 3.9
- CUDA支持 (推荐，用于GPU加速)

### 安装步骤

1. 克隆项目代码
```bash
git clone https://github.com/your-repo/face-attendance-system.git
cd face-attendance-system
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 启动应用
```bash
python app.py
```

4. 访问系统
在浏览器中访问 http://localhost:5000

## 目录结构

```
.
├── app.py                  # 主应用程序
├── face_services/          # 人脸识别服务
│   ├── baidu/              # 百度API集成
│   ├── face_recognition/   # Face_Recognition集成
│   ├── facenet/            # FaceNet集成
│   ├── hyp_oc/             # VGG-Face集成
│   └── modelscope/         # 活体检测与属性分析
├── static/                 # 静态资源
├── templates/              # HTML模板
├── class1/                 # 训练数据
├── temp_images/            # 临时图像
└── temp_exports/           # 导出文件
```

## 使用说明

1. **人脸注册**:
   - 将学生照片放入class1目录
   - 文件名格式: `学号-姓名.jpg/png`
   - 运行注册脚本 (详见各人脸服务目录下的register.py)

2. **考勤使用**:
   - 点击"人脸检测"开始摄像头
   - 选择识别方法进行人脸识别
   - 系统自动记录考勤信息

3. **考勤管理**:
   - 查看考勤记录
   - 按日期/姓名筛选
   - 导出为Excel

## 许可证

MIT License 

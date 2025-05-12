# 人脸识别系统 - 注册方法使用指南

本文档说明如何使用系统的四种人脸注册方法。所有方法现在都使用统一的流程，从 `members` 文件夹读取照片并注册用户。注册过程会自动从照片名称解析学号和姓名。

## 准备工作

1. 在项目根目录下，确保 `members` 文件夹存在
2. 将要注册的人脸照片放入 `members` 文件夹
3. 照片命名格式必须为: `学号-姓名.jpg`（或 `.png`），例如 `2022302181057-张三.jpg`

## 自动化功能

所有注册方法现在包含以下自动化功能：

1. **自动解析照片名称**：系统会从照片文件名中自动提取学号和姓名信息
2. **自动生成ID-姓名映射**：生成的映射会保存到所有服务目录中，无需手动编辑JSON文件
3. **自动更新学生数据库**：系统会自动更新VGG_Face使用的数据库文件

## 四种注册方法

### 1. 百度API注册

百度API注册会将照片上传到百度人脸库，需要网络连接。

```bash
python -m face_services.baidu.register
```

### 2. Face_Recognition注册

使用Face_Recognition库进行本地人脸特征提取和存储。

```bash
python -m face_services.face_recognition.register
```

### 3. FaceNet注册

使用FaceNet模型进行本地人脸特征提取和存储。

```bash
python -m face_services.facenet.register
```

### 4. VGG_Face注册

使用VGG_Face模型进行本地人脸特征提取和存储。

```bash
python -m face_services.hyp_oc.register
```

## 注意事项

1. 所有注册方法都从 `members` 文件夹读取照片
2. 照片必须是正面、清晰的人脸照片
3. 光线要充足，避免过暗或过亮
4. 照片中人脸占比要适中
5. 如果注册失败，请检查照片质量或尝试使用不同的照片
6. 注册成功后，用户信息和特征会保存在各自的数据库中

## 单个用户注册（仅限VGG_Face）

如果需要单独注册用户，VGG_Face模型还支持通过命令行参数直接注册单个用户：

```bash
python -m face_services.hyp_oc.student_register \
    --image "path/to/photo.jpg" \
    --student_id "2022302181057" \
    --name "张三" \
    --model_path "./face_services/hyp_oc/pretrained_weights/face_recognition_model.pth" \
    --pretrained_path "./face_services/hyp_oc/pretrained_weights/vgg_face_dag.pth" \
    --db_path "./face_services/hyp_oc/student_database.json" \
    --feature_path "./face_services/hyp_oc/student_features.pt" \
    --device "cpu" \
    --save_face
```

## 仅生成ID-姓名映射

如果只想生成ID-姓名映射文件，而不进行注册，可以使用以下命令：

```bash
python -m face_services.tools
```

这将读取`members`文件夹中的所有照片，生成ID到姓名的映射，并保存到所有服务目录中。

## 注册成功验证

注册成功后，可以使用系统的人脸识别功能进行验证测试。 
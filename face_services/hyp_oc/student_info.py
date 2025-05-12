import os
import json
import torch
import datetime

class StudentDatabase:
    """
    学生信息数据库，用于存储学生的基本信息和人脸特征
    """
    def __init__(self, db_path="student_database.json", feature_path="student_features.pt"):
        """
        初始化学生数据库
        
        参数:
            db_path: 学生基本信息的JSON文件路径
            feature_path: 学生人脸特征的PyTorch文件路径
        """
        self.db_path = db_path
        self.feature_path = feature_path
        self.students = {}  # 学号 -> 学生信息
        self.features = {}  # 学号 -> 人脸特征
        self.attendance = {}  # 日期 -> {学号 -> 考勤记录}
        
        # 加载数据库（如果存在）
        self.load_database()
    
    def load_database(self):
        """加载学生数据库"""
        # 加载学生基本信息
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.students = data.get('students', {})
                    self.attendance = data.get('attendance', {})
                print(f"成功加载学生数据库，共 {len(self.students)} 名学生")
            except Exception as e:
                print(f"加载学生数据库失败: {e}")
                self.students = {}
                self.attendance = {}
        
        # 加载人脸特征
        if os.path.exists(self.feature_path):
            try:
                data = torch.load(self.feature_path)
                self.features = data
                print(f"成功加载人脸特征，共 {len(self.features)} 名学生的特征")
            except Exception as e:
                print(f"加载人脸特征失败: {e}")
                self.features = {}
    
    def save_database(self):
        """保存学生数据库"""
        # 保存学生基本信息
        try:
            data = {
                'students': self.students,
                'attendance': self.attendance
            }
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 保存人脸特征
            torch.save(self.features, self.feature_path)
            print("学生数据库保存成功")
        except Exception as e:
            print(f"保存学生数据库失败: {e}")
    
    def add_student(self, student_id, name, features=None):
        """
        添加学生
        
        参数:
            student_id: 学号
            name: 姓名
            features: 人脸特征 (可选)
        
        返回:
            成功返回True，否则返回False
        """
        if student_id in self.students:
            print(f"学生 {student_id} 已存在")
            return False
        
        # 添加学生基本信息
        self.students[student_id] = {
            'name': name,
            'register_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果提供了特征，也保存特征
        if features is not None:
            self.features[student_id] = features
        
        # 保存数据库
        self.save_database()
        print(f"学生 {name}({student_id}) 添加成功")
        return True
    
    def update_student_feature(self, student_id, features):
        """
        更新学生的人脸特征
        
        参数:
            student_id: 学号
            features: 新的人脸特征
            
        返回:
            成功返回True，否则返回False
        """
        if student_id not in self.students:
            print(f"学生 {student_id} 不存在")
            return False
        
        # 更新特征
        self.features[student_id] = features
        
        # 保存数据库
        self.save_database()
        student_name = self.students[student_id]['name']
        print(f"学生 {student_name}({student_id}) 的人脸特征更新成功")
        return True
    
    def record_attendance(self, student_id, is_present=True):
        """
        记录学生考勤
        
        参数:
            student_id: 学号
            is_present: 是否出勤
            
        返回:
            成功返回True，否则返回False
        """
        if student_id not in self.students:
            print(f"学生 {student_id} 不存在")
            return False
        
        # 获取当前日期
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 确保当天记录存在
        if today not in self.attendance:
            self.attendance[today] = {}
        
        # 记录考勤
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        self.attendance[today][student_id] = {
            'time': current_time,
            'present': is_present
        }
        
        # 保存数据库
        self.save_database()
        student_name = self.students[student_id]['name']
        status = "出勤" if is_present else "缺勤"
        print(f"学生 {student_name}({student_id}) 在 {today} {current_time} {status}")
        return True
    
    def get_student_info(self, student_id):
        """
        获取学生信息
        
        参数:
            student_id: 学号
            
        返回:
            学生信息字典，如果不存在则返回None
        """
        return self.students.get(student_id)
    
    def get_student_feature(self, student_id):
        """
        获取学生人脸特征
        
        参数:
            student_id: 学号
            
        返回:
            学生人脸特征，如果不存在则返回None
        """
        return self.features.get(student_id)
    
    def get_all_students(self):
        """
        获取所有学生信息
        
        返回:
            所有学生信息的字典
        """
        return self.students
    
    def get_all_features(self):
        """
        获取所有学生的人脸特征
        
        返回:
            所有学生特征的字典 {学号: 特征}
        """
        return self.features
    
    def get_attendance(self, date=None):
        """
        获取考勤记录
        
        参数:
            date: 日期，格式为'YYYY-MM-DD'，如果为None则返回当天记录
            
        返回:
            考勤记录字典
        """
        if date is None:
            date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        return self.attendance.get(date, {}) 
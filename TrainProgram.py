# -*- coding: utf-8 -*-
"""
基于YOLOv8的模型训练程序
TrainProgram.py - 训练模型的主程序文件

功能：
1. 上传图片或文件夹作为数据集
2. 修改训练配置参数
3. 启动模型训练
4. 显示训练进度
"""

import os
import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QProgressBar, QLabel
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication
from ultralytics import YOLO
import torch
import shutil
from UIProgram.TrainUiMain import Ui_TrainMainWindow
from UIProgram.QssLoader import QSSLoader
import Config

class TrainThread(QThread):
    """训练线程类，用于后台执行训练任务"""
    # 声明信号
    update_progress = pyqtSignal(int, str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, data_path, model_type, epochs, batch, lr, img_size, device):
        """初始化训练线程
        
        Args:
            data_path: 数据集路径
            model_type: 模型类型
            epochs: 训练轮数
            batch: 批次大小
            lr: 学习率
            img_size: 图片大小
            device: 设备选择
        """
        super(TrainThread, self).__init__()
        self.data_path = data_path
        self.model_type = model_type
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.img_size = img_size
        self.device = device
        self.is_running = True
    
    def run(self):
        """线程运行方法，执行训练任务"""
        try:
            # 加载预训练模型
            self.update_progress.emit(0, "加载预训练模型...")
            
            # 检查是否为自定义模型
            if self.model_type == 'YOLOv8n-ASA':
                from models.yolov8n_asa import YOLOv8nASA
                model = YOLOv8nASA()
            elif self.model_type == 'YOLOv8n-FGP':
                from models.yolov8n_fgp import YOLOv8nFGP
                model = YOLOv8nFGP()
            else:
                # 使用标准YOLO模型
                model = YOLO(self.model_type)
            
            # 准备训练参数
            train_args = {
                'data': self.data_path,
                'epochs': self.epochs,
                'batch': self.batch,
                'lr0': self.lr,
                'imgsz': self.img_size,
                'project': 'runs/train',
                'name': f'train_{int(time.time())}',
                'exist_ok': True
            }
            
            # 安全地设置设备 - 如果是'0'则不明确指定，让系统自动选择
            if self.device != '0':
                train_args['device'] = self.device
            
            self.update_progress.emit(5, "开始训练...")
            
            # 开始训练
            results = model.train(**train_args)
            
            self.update_progress.emit(100, "训练完成！")
            self.training_finished.emit()
        except Exception as e:
            self.training_error.emit(f"训练过程中出错: {str(e)}")
    
    def stop(self):
        """停止训练线程"""
        self.is_running = False

class TrainMainWindow(QMainWindow):
    """训练模型的主窗口类"""
    
    def __init__(self, parent=None):
        """初始化主窗口"""
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_TrainMainWindow()
        self.ui.setupUi(self)
        self.initUI()
        self.signalconnect()
        
        # 加载CSS样式文件，美化界面
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)
    
    def initUI(self):
        """初始化UI界面"""
        # 设置默认值
        self.ui.spinBox_epochs.setValue(100)
        self.ui.spinBox_batch.setValue(8)
        self.ui.lineEdit_lr.setText("0.01")
        self.ui.spinBox_img_size.setValue(640)
        self.ui.spinBox_val_interval.setValue(1)
        
        # 填充模型类型下拉框
        model_types = [
            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 
            'yolov8l.pt', 'yolov8x.pt',
            'YOLOv8n-ASA', 'YOLOv8n-FGP'
        ]
        self.ui.comboBox_model_type.addItems(model_types)
        
        # 填充设备选择下拉框
        devices = ['cpu', '0']  # 始终显示CPU和GPU选项
        self.ui.comboBox_device.addItems(devices)
        
        # 设置默认数据集路径
        default_data_path = os.path.join(os.path.dirname(__file__), 'datasets', 'helmetData', 'data.yaml')
        self.ui.lineEdit_data_path.setText(default_data_path)
    
    def signalconnect(self):
        """连接信号和槽"""
        self.ui.pushButton_browse.clicked.connect(self.browse_data_path)
        self.ui.pushButton_upload_images.clicked.connect(self.upload_images)
        self.ui.pushButton_upload_folder.clicked.connect(self.upload_folder)
        self.ui.pushButton_train.clicked.connect(self.start_training)
        self.ui.pushButton_cancel.clicked.connect(QCoreApplication.quit)
    
    def browse_data_path(self):
        """浏览数据集路径"""
        file_path, _ = QFileDialog.getOpenFileName(
            None, '选择数据集配置文件', 
            os.path.join(os.path.dirname(__file__), 'datasets'), 
            "YAML files (*.yaml)"
        )
        if file_path:
            self.ui.lineEdit_data_path.setText(file_path)
    
    def upload_images(self):
        """上传图片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            None, '选择图片', 
            os.path.join(os.path.dirname(__file__), 'TestFiles'), 
            "Image files (*.jpg *.jpeg *.png)"
        )
        if file_paths:
            # 选择保存目录
            save_dir = QFileDialog.getExistingDirectory(
                None, '选择保存目录', 
                os.path.join(os.path.dirname(__file__), 'datasets')
            )
            if save_dir:
                # 确保目录存在
                images_dir = os.path.join(save_dir, 'images')
                labels_dir = os.path.join(save_dir, 'labels')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                
                # 复制图片到目录
                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(images_dir, file_name)
                    shutil.copyfile(file_path, dest_path)
                
                QMessageBox.about(self, '提示', f'成功上传 {len(file_paths)} 张图片！')
    
    def upload_folder(self):
        """上传文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            None, '选择图片文件夹', 
            os.path.join(os.path.dirname(__file__), 'TestFiles')
        )
        if folder_path:
            # 选择保存目录
            save_dir = QFileDialog.getExistingDirectory(
                None, '选择保存目录', 
                os.path.join(os.path.dirname(__file__), 'datasets')
            )
            if save_dir:
                # 确保目录存在
                images_dir = os.path.join(save_dir, 'images')
                labels_dir = os.path.join(save_dir, 'labels')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                
                # 复制图片到目录
                img_count = 0
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        dest_path = os.path.join(images_dir, file_name)
                        shutil.copyfile(file_path, dest_path)
                        img_count += 1
                
                QMessageBox.about(self, '提示', f'成功上传 {img_count} 张图片！')
    
    def start_training(self):
        """开始训练"""
        # 获取训练参数
        data_path = self.ui.lineEdit_data_path.text()
        model_type = self.ui.comboBox_model_type.currentText()
        epochs = self.ui.spinBox_epochs.value()
        batch = self.ui.spinBox_batch.value()
        
        try:
            lr = float(self.ui.lineEdit_lr.text())
        except ValueError:
            QMessageBox.warning(self, '错误', '学习率必须是数字！')
            return
        
        img_size = self.ui.spinBox_img_size.value()
        device = self.ui.comboBox_device.currentText()
        
        # 验证参数
        if not os.path.exists(data_path):
            QMessageBox.warning(self, '错误', '数据集路径不存在！')
            return
        
        # 禁用训练按钮
        self.ui.pushButton_train.setEnabled(False)
        
        # 创建训练线程
        self.train_thread = TrainThread(
            data_path, model_type, epochs, batch, lr, img_size, device
        )
        
        # 连接信号
        self.train_thread.update_progress.connect(self.update_progress)
        self.train_thread.training_finished.connect(self.training_finished)
        self.train_thread.training_error.connect(self.training_error)
        
        # 启动线程
        self.train_thread.start()
    
    def update_progress(self, value, status):
        """更新训练进度"""
        self.ui.progressBar.setValue(value)
        self.ui.label_status.setText(f"状态: {status}")
        QApplication.processEvents()
    
    def training_finished(self):
        """训练完成"""
        self.ui.pushButton_train.setEnabled(True)
        QMessageBox.about(self, '提示', '训练完成！模型已保存到 runs/train 目录。')
    
    def training_error(self, error_msg):
        """训练错误"""
        self.ui.pushButton_train.setEnabled(True)
        QMessageBox.critical(self, '错误', error_msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TrainMainWindow()
    win.show()
    sys.exit(app.exec_())

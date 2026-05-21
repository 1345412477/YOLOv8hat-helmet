# -*- coding: utf-8 -*-
"""
基于YOLOv8的安全帽检查与识别系统
MainProgram.py - 主程序文件

功能：
1. 加载YOLOv8模型进行安全帽检测
2. 提供图形界面，支持图片、视频和摄像头输入
3. 显示检测结果和目标位置信息
4. 支持批量处理和结果保存
"""

# 导入必要的库
import os
import sys
# 添加UIProgram目录到Python路径，确保能正确导入UI模块
sys.path.append('UIProgram')
import time
from PIL import ImageFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QHeaderView, QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication
from ultralytics import YOLO  # YOLOv8模型库
import detect_tools as tools  # 自定义工具函数
import cv2  # 图像处理库
import Config  # 配置文件
from UIProgram.QssLoader import QSSLoader  # CSS加载器
from UIProgram.precess_bar import ProgressBar  # 进度条组件
from UIProgram.loading_animation import LoadingAnimation  # 加载动画组件
from UIProgram.UiMain import Ui_MainWindow  # UI界面类
import numpy as np  # 数值计算库
import torch  # PyTorch库
import shutil  # 文件操作库

class TrainThread(QThread):
    """训练线程类，用于后台执行训练任务"""
    # 声明信号
    update_progress = pyqtSignal(int, str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, data_path, model_type, epochs, batch, lr, img_size, device, val_interval):
        """初始化训练线程
        
        Args:
            data_path: 数据集路径
            model_type: 模型类型
            epochs: 训练轮数
            batch: 批次大小
            lr: 学习率
            img_size: 图片大小
            device: 设备选择
            val_interval: 验证间隔
        """
        super(TrainThread, self).__init__()
        self.data_path = data_path
        self.model_type = model_type
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.img_size = img_size
        self.device = device
        self.val_interval = val_interval
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

class MainWindow(QMainWindow):
    """主窗口类，负责整个应用的界面和逻辑"""
    
    def __init__(self, parent=None):
        """初始化主窗口"""
        super(QMainWindow, self).__init__(parent)
        # 初始化UI界面
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # 初始化主窗口设置
        self.initMain()
        # 初始化训练相关设置
        self.initTrain()
        # 连接信号和槽
        self.signalconnect()

        # 加载CSS样式文件，美化界面
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

    def signalconnect(self):
        """连接信号和槽，绑定按钮点击事件"""
        # 检测相关信号
        self.ui.PicBtn.clicked.connect(self.open_img)  # 图片按钮点击事件
        self.ui.comboBox.activated.connect(self.combox_change)  # 下拉框选择事件
        self.ui.VideoBtn.clicked.connect(self.vedio_show)  # 视频按钮点击事件
        self.ui.CapBtn.clicked.connect(self.camera_show)  # 摄像头按钮点击事件
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)  # 保存按钮点击事件
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)  # 退出按钮点击事件
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)  # 批量处理按钮点击事件
        self.ui.comboBox_model.currentIndexChanged.connect(self.model_change)  # 模型选择事件
        
        # 训练相关信号
        self.ui.pushButton_browse.clicked.connect(self.browse_data_path)  # 浏览数据集路径
        self.ui.pushButton_upload_images.clicked.connect(self.upload_images)  # 上传图片
        self.ui.pushButton_upload_folder.clicked.connect(self.upload_folder)  # 上传文件夹
        self.ui.pushButton_train.clicked.connect(self.start_training)  # 开始训练



    def initMain(self):
        """初始化主窗口设置和变量"""
        # 显示窗口尺寸
        self.show_width = 770
        self.show_height = 480

        # 图片相关变量
        self.org_path = None  # 原始图片路径
        self.draw_img = None  # 绘制后的图片
        self.results = None  # 检测结果
        self.location_list = []  # 目标位置列表
        self.cls_list = []  # 目标类别列表
        self.conf_list = []  # 目标置信度列表
        self.img_width = 0  # 图片宽度
        self.img_height = 0  # 图片高度

        # 摄像头相关变量
        self.is_camera_open = False  # 摄像头是否打开
        self.cap = None  # 摄像头捕获对象

        # 设备选择（GPU或CPU）
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 动态扫描 models 文件夹中的模型文件
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        else:
            model_files = []
        
        # 添加自定义模型选项
        custom_models = ['YOLOv8n-ASA', 'YOLOv8n-FGP']
        
        if not model_files and not custom_models:
            QMessageBox.critical(self, '错误', 'models 文件夹中没有找到模型文件！')
            QMessageBox.information(self, '提示', '请在models目录中放置YOLOv8模型文件（.pt格式）')
            sys.exit(1)
        
        # 初始化模型选择下拉框
        all_models = model_files + custom_models
        self.ui.comboBox_model.addItems(all_models)
        self.ui.comboBox_model.setCurrentIndex(0)

        # 加载检测模型
        try:
            first_model_name = all_models[0]
            if first_model_name in custom_models:
                # 加载自定义模型
                if first_model_name == 'YOLOv8n-ASA':
                    from models.yolov8n_asa import YOLOv8nASA
                    self.model = YOLOv8nASA()
                else:  # YOLOv8n-FGP
                    from models.yolov8n_fgp import YOLOv8nFGP
                    self.model = YOLOv8nFGP()
            else:
                # 加载标准YOLO模型
                first_model_path = os.path.join(models_dir, first_model_name)
                self.model = YOLO(first_model_path, task='detect')  # 加载YOLOv8模型
            
            self.model(np.zeros((48, 48, 3)))  # 预先加载推理模型，加速后续检测
            self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)  # 加载字体
            self.colors = tools.Colors()  # 用于绘制不同颜色矩形框
        except Exception as e:
            QMessageBox.critical(self, '模型加载错误', f'无法加载模型: {str(e)}\n请检查模型文件是否损坏或路径是否正确。')
            sys.exit(1)

        # 定时器初始化
        self.timer_camera = QTimer()  # 用于更新视频图像
        self.timer_save_video = QTimer()  # 用于保存视频

        # 加载动画初始化
        self.loading_animation = LoadingAnimation(self)

        # 表格设置
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)  # 垂直表头固定
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)  # 默认行高
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 水平表头自动调整
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替
        self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑

    def initTrain(self):
        """初始化训练相关设置"""
        # 设置训练默认值
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
        devices = ['CPU', 'GPU']  # 始终显示CPU和GPU选项
        self.ui.comboBox_device.addItems(devices)
        
        # 设置默认数据集路径
        default_data_path = os.path.join(os.path.dirname(__file__), 'datasets', 'helmetData', 'data.yaml')
        self.ui.lineEdit_data_path.setText(default_data_path)

    def process_detection_results(self, results, img_path):
        """处理检测结果并更新UI"""
        location_list = results.boxes.xyxy.tolist()
        location_list = [list(map(int, e)) for e in location_list]
        cls_list = results.boxes.cls.tolist()
        cls_list = [int(i) for i in cls_list]
        conf_list = results.boxes.conf.tolist()
        
        # 不使用过滤，直接使用原始检测结果
        # location_list, cls_list, conf_list = self.filter_overlapping_persons(
        #     location_list, cls_list, conf_list
        # )
        
        self.location_list = location_list
        self.cls_list = cls_list
        self.conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

        # 使用原始的检测结果
        now_img = results.plot()
        
        self.draw_img = now_img
        
        self.update_ui_with_results(now_img, img_path)
        
        return now_img

    def update_ui_with_results(self, img, img_path):
        """使用检测结果更新UI"""
        # 更新显示窗口尺寸为label_show组件的实际尺寸
        self.show_width = self.ui.label_show.width()
        self.show_height = self.ui.label_show.height()
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(img)
        # 使用高质量的LANCZOS4插值方法，在缩小图片时效果更好
        resize_cvimg = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LANCZOS4)
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        
        # 设置路径显示
        self.ui.PiclineEdit.setText(img_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
        choose_list.extend(target_names)

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

        # 更新目标信息
        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
            # 设置坐标位置值
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 更新表格
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)

    def open_img(self):
        """打开并处理图片"""
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            None, '打开图片', 
            os.path.join(os.path.dirname(__file__), 'TestFiles'), 
            "Image files (*.jpg *.jpeg *.png)"
        )
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        
        try:
            # 显示加载动画
            self.loading_animation.setText("正在检测...")
            self.loading_animation.start()
            QApplication.processEvents()
            
            # 目标检测
            t1 = time.time()
            self.results = self.model(self.org_path)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            # 处理检测结果
            self.process_detection_results(self.results, self.org_path)
        except Exception as e:
            QMessageBox.critical(self, '图片处理错误', f'处理图片时发生错误: {str(e)}\n请确保图片文件格式正确且未损坏。')
            import traceback
            traceback.print_exc()
        finally:
            # 停止加载动画
            self.loading_animation.stop()

    def detact_batch_imgs(self):
        """批量处理图片"""
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None
        
        directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        if not directory:
            return
        
        self.org_path = directory
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        
        try:
            for file_name in os.listdir(directory):
                full_path = os.path.join(directory, file_name)
                if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                    # 目标检测
                    t1 = time.time()
                    self.results = self.model(full_path)[0]
                    t2 = time.time()
                    take_time_str = '{:.3f} s'.format(t2 - t1)
                    self.ui.time_lb.setText(take_time_str)

                    # 处理检测结果
                    self.process_detection_results(self.results, full_path)
                    
                    self.ui.tableWidget.scrollToBottom()
                    QApplication.processEvents()  # 刷新页面
        except Exception as e:
            QMessageBox.critical(self, '错误', f'批量处理失败: {str(e)}')

    def combox_change(self):
        """处理下拉框变化"""
        com_text = self.ui.comboBox.currentText()
        if com_text == '全部':
            cur_box = self.location_list
            cur_img = self.results.plot()
            if self.cls_list:
                self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.ui.label_conf.setText(str(self.conf_list[0]))
        else:
            try:
                index = int(com_text.split('_')[-1])
                if 0 <= index < len(self.location_list):
                    cur_box = [self.location_list[index]]
                    cur_img = self.results[index].plot()
                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[index]])
                    self.ui.label_conf.setText(str(self.conf_list[index]))
                else:
                    return
            except (ValueError, IndexError):
                return

        # 设置坐标位置值
        if cur_box:
            self.ui.label_xmin.setText(str(cur_box[0][0]))
            self.ui.label_ymin.setText(str(cur_box[0][1]))
            self.ui.label_xmax.setText(str(cur_box[0][2]))
            self.ui.label_ymax.setText(str(cur_box[0][3]))

            # 使用高质量的LANCZOS4插值方法，在缩小图片时效果更好
            resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height), interpolation=cv2.INTER_LANCZOS4)
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.clear()
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

    def model_change(self):
        """切换检测模型"""
        model_name = self.ui.comboBox_model.currentText()
        custom_models = ['YOLOv8n-ASA', 'YOLOv8n-FGP']
        
        try:
            self.loading_animation.setText("正在加载模型...")
            self.loading_animation.start()
            QApplication.processEvents()
            
            if model_name in custom_models:
                # 加载自定义模型
                if model_name == 'YOLOv8n-ASA':
                    from models.yolov8n_asa import YOLOv8nASA
                    self.model = YOLOv8nASA()
                else:  # YOLOv8n-FGP
                    from models.yolov8n_fgp import YOLOv8nFGP
                    self.model = YOLOv8nFGP()
            else:
                # 加载标准YOLO模型
                model_path = os.path.join('models', model_name)
                self.model = YOLO(model_path, task='detect')
            
            self.model(np.zeros((48, 48, 3)))
            
            self.loading_animation.stop()
            QMessageBox.information(self, '模型切换', f'已成功切换到模型: {model_name}')
        except Exception as e:
            self.loading_animation.stop()
            QMessageBox.critical(self, '模型加载错误', f'无法加载模型 {model_name}: {str(e)}')

    def get_video_path(self):
        """获取视频路径"""
        file_path, _ = QFileDialog.getOpenFileName(
            None, '打开视频', 
            os.path.join(os.path.dirname(__file__), 'TestFiles'), 
            "Video files (*.avi *.mp4)"
        )
        if not file_path:
            return None
        self.org_path = file_path
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        """开始视频处理"""
        # 清空表格
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 清空下拉框
        self.ui.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(33)  # 约30fps
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, path=None):
        """显示表格信息"""
        # 禁用排序和自动调整大小以提高性能
        self.ui.tableWidget.setSortingEnabled(False)
        self.ui.tableWidget.setUpdatesEnabled(False)
        
        try:
            # 计算需要的行数
            total_rows = len(locations)
            current_rows = self.ui.tableWidget.rowCount()
            
            # 如果需要更多行，添加它们
            if total_rows > current_rows:
                self.ui.tableWidget.setRowCount(total_rows)
            
            # 批量更新表格数据
            for i, (location, cls, conf) in enumerate(zip(locations, clses, confs)):
                item_id = QTableWidgetItem(str(i + 1))  # 序号
                item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
                item_path = QTableWidgetItem(str(path))  # 路径

                item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
                item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

                item_conf = QTableWidgetItem(str(conf))
                item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

                item_location = QTableWidgetItem(str(location))  # 目标框位置

                self.ui.tableWidget.setItem(i, 0, item_id)
                self.ui.tableWidget.setItem(i, 1, item_path)
                self.ui.tableWidget.setItem(i, 2, item_cls)
                self.ui.tableWidget.setItem(i, 3, item_conf)
                self.ui.tableWidget.setItem(i, 4, item_location)
            
            # 如果需要更少行，删除多余的
            if total_rows < current_rows:
                for i in range(current_rows - 1, total_rows - 1, -1):
                    self.ui.tableWidget.removeRow(i)
        finally:
            # 重新启用排序和自动调整大小
            self.ui.tableWidget.setUpdatesEnabled(True)
            self.ui.tableWidget.setSortingEnabled(True)
            self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        """停止视频处理"""
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        self.timer_camera.stop()

    def open_frame(self):
        """处理视频帧"""
        if not self.cap:
            return
            
        ret, now_img = self.cap.read()
        if ret:
            try:
                t1 = time.time()
                results = self.model(now_img)[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                conf_list = results.boxes.conf.tolist()
                
                # 不使用过滤，直接使用原始检测结果
                # location_list, cls_list, conf_list = self.filter_overlapping_persons(
                #     location_list, cls_list, conf_list
                # )
                
                self.location_list = location_list
                self.cls_list = cls_list
                self.conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

                # 使用原始的检测结果
                now_img = results.plot()

                # 更新显示窗口尺寸为label_show组件的实际尺寸
                self.show_width = self.ui.label_show.width()
                self.show_height = self.ui.label_show.height()
                self.img_width, self.img_height = self.get_resize_size(now_img)
                # 使用高质量的LANCZOS4插值方法，在缩小图片时效果更好
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height), interpolation=cv2.INTER_LANCZOS4)
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)

                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))

                choose_list = ['全部']
                target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
                choose_list.extend(target_names)

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)

                if target_nums >= 1:
                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                    self.ui.label_conf.setText(str(self.conf_list[0]))
                    self.ui.label_xmin.setText(str(self.location_list[0][0]))
                    self.ui.label_ymin.setText(str(self.location_list[0][1]))
                    self.ui.label_xmax.setText(str(self.location_list[0][2]))
                    self.ui.label_ymax.setText(str(self.location_list[0][3]))
                else:
                    self.ui.type_lb.setText('')
                    self.ui.label_conf.setText('')
                    self.ui.label_xmin.setText('')
                    self.ui.label_ymin.setText('')
                    self.ui.label_xmax.setText('')
                    self.ui.label_ymax.setText('')

                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
            except Exception as e:
                print(f'处理视频帧时出错: {str(e)}')

        else:
            self.video_stop()

    def vedio_show(self):
        """显示视频并进行实时检测"""
        # 如果摄像头已打开，先关闭
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')

        # 获取视频文件路径
        video_path = self.get_video_path()
        if not video_path:
            return
            
        try:
            # 打开视频文件
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, '错误', '无法打开视频文件，请检查文件格式是否正确')
                return
            # 开始视频处理
            self.video_start()
            # 禁用下拉框，因为视频模式下不需要选择目标
            self.ui.comboBox.setDisabled(True)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'打开视频失败: {str(e)}\n请确保视频文件格式正确且未损坏')

    def camera_show(self):
        """显示摄像头并进行实时检测"""
        # 切换摄像头状态
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            try:
                # 更新摄像头状态显示
                self.ui.CaplineEdit.setText('摄像头开启')
                # 打开摄像头（索引为0的默认摄像头）
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    QMessageBox.critical(self, '错误', '无法打开摄像头，请检查摄像头是否连接正常')
                    self.is_camera_open = False
                    self.ui.CaplineEdit.setText('摄像头未开启')
                    return
                # 开始视频处理
                self.video_start()
                # 禁用下拉框，因为摄像头模式下不需要选择目标
                self.ui.comboBox.setDisabled(True)
            except Exception as e:
                QMessageBox.critical(self, '错误', f'打开摄像头失败: {str(e)}\n请检查摄像头是否连接正常或被其他程序占用')
                self.is_camera_open = False
                self.ui.CaplineEdit.setText('摄像头未开启')
        else:
            # 关闭摄像头
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.video_stop()  # 停止视频处理
            self.ui.label_show.clear()
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    print(f'释放摄像头时出错: {str(e)}')

    def get_resize_size(self, img):
        """获取调整后的图片尺寸，保持宽高比"""
        _img = img.copy()
        img_height, img_width, _ = _img.shape
        ratio = img_width / img_height
        
        # 根据显示窗口尺寸计算调整后的图片尺寸
        if ratio >= self.show_width / self.show_height:
            # 以宽度为基准
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            # 以高度为基准
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        
        return self.img_width, self.img_height
    
    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super(MainWindow, self).resizeEvent(event)
        
        # 更新显示窗口尺寸
        self.show_width = self.ui.label_show.width()
        self.show_height = self.ui.label_show.height()
        
        # 如果有绘制的图片，重新调整大小并显示
        if hasattr(self, 'draw_img') and self.draw_img is not None:
            try:
                self.img_width, self.img_height = self.get_resize_size(self.draw_img)
                # 使用高质量的LANCZOS4插值方法，在缩小图片时效果更好
                resize_cvimg = cv2.resize(self.draw_img, (self.img_width, self.img_height), interpolation=cv2.INTER_LANCZOS4)
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)
            except Exception as e:
                pass

    def save_detect_video(self):
        """保存检测结果"""
        # 检查是否有可保存的内容
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        # 摄像头视频无法保存
        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        # 处理视频保存
        if self.cap:
            res = QMessageBox.information(
                self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if res == QMessageBox.Yes:
                # 停止视频播放
                self.video_stop()
                # 获取当前选中的目标
                com_text = self.ui.comboBox.currentText()
                # 创建并启动线程处理视频保存
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text)
                self.btn2Thread_object.start()
                # 连接进度更新信号
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
        else:
            # 处理图片保存
            try:
                # 确保保存目录存在
                if not os.path.exists(Config.save_path):
                    os.makedirs(Config.save_path)
                    
                if os.path.isfile(self.org_path):
                    # 保存单个图片
                    if self.draw_img is None:
                        QMessageBox.critical(self, '错误', '没有可保存的检测结果，请先进行检测')
                        return
                    fileName = os.path.basename(self.org_path)
                    name, end_name = fileName.rsplit(".", 1)
                    save_name = name + '_detect_result.' + end_name
                    save_img_path = os.path.join(Config.save_path, save_name)
                    # 保存图片
                    cv2.imwrite(save_img_path, self.draw_img)
                    QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
                else:
                    # 保存批量图片
                    img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                    saved_count = 0
                    for file_name in os.listdir(self.org_path):
                        full_path = os.path.join(self.org_path, file_name)
                        if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                            try:
                                name, end_name = file_name.rsplit(".", 1)
                                save_name = name + '_detect_result.' + end_name
                                save_img_path = os.path.join(Config.save_path, save_name)
                                # 重新检测图片
                                results = self.model(full_path)[0]
                                now_img = results.plot()
                                # 保存图片
                                cv2.imwrite(save_img_path, now_img)
                                saved_count += 1
                            except Exception as img_error:
                                print(f'保存图片 {file_name} 时出错: {str(img_error)}')

                    QMessageBox.about(self, '提示', f'图片保存成功!\n共保存 {saved_count} 张图片\n文件路径:{Config.save_path}')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存失败: {str(e)}\n请检查保存路径是否有写入权限')

    def update_process_bar(self, cur_num, total):
        """更新进度条，显示保存视频的进度"""
        if cur_num == 1:
            # 初始化并显示进度条
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        
        if cur_num >= total:
            # 保存完成，关闭进度条并显示提示
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        
        if hasattr(self, 'progress_bar') and not self.progress_bar.isVisible():
            # 点击取消保存时，终止进程
            if hasattr(self, 'btn2Thread_object'):
                self.btn2Thread_object.stop()
            return
        
        # 计算并更新进度值
        value = int(cur_num / total * 100)
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(cur_num, total, value)
            QApplication.processEvents()  # 刷新界面

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
        val_interval = self.ui.spinBox_val_interval.value()
        
        # 验证参数
        if not os.path.exists(data_path):
            QMessageBox.warning(self, '错误', '数据集路径不存在！')
            return
        
        # 禁用训练按钮
        self.ui.pushButton_train.setEnabled(False)
        
        # 创建训练线程
        self.train_thread = TrainThread(
            data_path, model_type, epochs, batch, lr, img_size, device, val_interval
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


class btn2Thread(QThread):
    """视频保存线程类，用于后台处理视频检测和保存"""
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path, model, com_text):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        self.is_running = True



    def run(self):
        """线程运行方法，处理视频检测和保存"""
        cap = cv2.VideoCapture(self.org_path)
        if not cap.isOpened():
            self.update_ui_signal.emit(1, 1)
            return
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            
            fileName = os.path.basename(self.org_path)
            name, end_name = fileName.split('.')
            save_name = name + '_detect_result.avi'
            
            if not os.path.exists(Config.save_path):
                os.makedirs(Config.save_path)
                
            save_video_path = os.path.join(Config.save_path, save_name)
            out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

            prop = cv2.CAP_PROP_FRAME_COUNT
            total = int(cap.get(prop))
            print("[INFO] 视频总帧数：{}".format(total))
            cur_num = 0

            while (cap.isOpened() and self.is_running):
                cur_num += 1
                print('当前第{}帧，总帧数{}'.format(cur_num, total))
                ret, frame = cap.read()
                if ret:
                    results = self.model(frame)[0]
                    
                    location_list = results.boxes.xyxy.tolist()
                    location_list = [list(map(int, e)) for e in location_list]
                    cls_list = results.boxes.cls.tolist()
                    cls_list = [int(i) for i in cls_list]
                    conf_list = results.boxes.conf.tolist()
                    
                    # 不使用过滤，直接使用原始检测结果
                    # location_list, cls_list, conf_list = self.filter_overlapping_persons(
                    #     location_list, cls_list, conf_list
                    # )
                    
                    # 使用原始的检测结果
                    frame = results.plot()
                    
                    out.write(frame)
                    self.update_ui_signal.emit(cur_num, total)
                else:
                    break
        except Exception as e:
            print(f'保存视频时出错: {str(e)}')
        finally:
            # 释放资源
            try:
                cap.release()
                out.release()
            except:
                pass

    def stop(self):
        """停止线程"""
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-
"""
训练模型的UI界面
"""

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TrainMainWindow(object):
    def setupUi(self, TrainMainWindow):
        TrainMainWindow.setObjectName("TrainMainWindow")
        TrainMainWindow.resize(800, 600)
        TrainMainWindow.setMinimumSize(QtCore.QSize(800, 600))
        TrainMainWindow.setMaximumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(TrainMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 标题标签
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(20, 10, 760, 40))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")
        
        # 数据集部分
        self.groupBox_data = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_data.setGeometry(QtCore.QRect(20, 60, 760, 120))
        self.groupBox_data.setObjectName("groupBox_data")
        
        # 数据集路径标签
        self.label_data_path = QtWidgets.QLabel(self.groupBox_data)
        self.label_data_path.setGeometry(QtCore.QRect(20, 30, 100, 20))
        self.label_data_path.setObjectName("label_data_path")
        
        # 数据集路径输入框
        self.lineEdit_data_path = QtWidgets.QLineEdit(self.groupBox_data)
        self.lineEdit_data_path.setGeometry(QtCore.QRect(120, 30, 450, 20))
        self.lineEdit_data_path.setObjectName("lineEdit_data_path")
        
        # 浏览按钮
        self.pushButton_browse = QtWidgets.QPushButton(self.groupBox_data)
        self.pushButton_browse.setGeometry(QtCore.QRect(580, 30, 100, 20))
        self.pushButton_browse.setObjectName("pushButton_browse")
        
        # 上传图片按钮
        self.pushButton_upload_images = QtWidgets.QPushButton(self.groupBox_data)
        self.pushButton_upload_images.setGeometry(QtCore.QRect(120, 70, 150, 30))
        self.pushButton_upload_images.setObjectName("pushButton_upload_images")
        
        # 上传文件夹按钮
        self.pushButton_upload_folder = QtWidgets.QPushButton(self.groupBox_data)
        self.pushButton_upload_folder.setGeometry(QtCore.QRect(300, 70, 150, 30))
        self.pushButton_upload_folder.setObjectName("pushButton_upload_folder")
        
        # 配置文件部分
        self.groupBox_config = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_config.setGeometry(QtCore.QRect(20, 200, 760, 200))
        self.groupBox_config.setObjectName("groupBox_config")
        
        # 模型类型标签
        self.label_model_type = QtWidgets.QLabel(self.groupBox_config)
        self.label_model_type.setGeometry(QtCore.QRect(20, 30, 100, 20))
        self.label_model_type.setObjectName("label_model_type")
        
        # 模型类型下拉框
        self.comboBox_model_type = QtWidgets.QComboBox(self.groupBox_config)
        self.comboBox_model_type.setGeometry(QtCore.QRect(120, 30, 200, 20))
        self.comboBox_model_type.setObjectName("comboBox_model_type")
        
        # 训练轮数标签
        self.label_epochs = QtWidgets.QLabel(self.groupBox_config)
        self.label_epochs.setGeometry(QtCore.QRect(350, 30, 100, 20))
        self.label_epochs.setObjectName("label_epochs")
        
        # 训练轮数输入框
        self.spinBox_epochs = QtWidgets.QSpinBox(self.groupBox_config)
        self.spinBox_epochs.setGeometry(QtCore.QRect(450, 30, 100, 20))
        self.spinBox_epochs.setMinimum(1)
        self.spinBox_epochs.setMaximum(1000)
        self.spinBox_epochs.setObjectName("spinBox_epochs")
        
        # 批次大小标签
        self.label_batch = QtWidgets.QLabel(self.groupBox_config)
        self.label_batch.setGeometry(QtCore.QRect(20, 70, 100, 20))
        self.label_batch.setObjectName("label_batch")
        
        # 批次大小输入框
        self.spinBox_batch = QtWidgets.QSpinBox(self.groupBox_config)
        self.spinBox_batch.setGeometry(QtCore.QRect(120, 70, 100, 20))
        self.spinBox_batch.setMinimum(1)
        self.spinBox_batch.setMaximum(128)
        self.spinBox_batch.setObjectName("spinBox_batch")
        
        # 学习率标签
        self.label_lr = QtWidgets.QLabel(self.groupBox_config)
        self.label_lr.setGeometry(QtCore.QRect(250, 70, 100, 20))
        self.label_lr.setObjectName("label_lr")
        
        # 学习率输入框
        self.lineEdit_lr = QtWidgets.QLineEdit(self.groupBox_config)
        self.lineEdit_lr.setGeometry(QtCore.QRect(350, 70, 100, 20))
        self.lineEdit_lr.setObjectName("lineEdit_lr")
        
        # 图片大小标签
        self.label_img_size = QtWidgets.QLabel(self.groupBox_config)
        self.label_img_size.setGeometry(QtCore.QRect(480, 70, 100, 20))
        self.label_img_size.setObjectName("label_img_size")
        
        # 图片大小输入框
        self.spinBox_img_size = QtWidgets.QSpinBox(self.groupBox_config)
        self.spinBox_img_size.setGeometry(QtCore.QRect(580, 70, 100, 20))
        self.spinBox_img_size.setMinimum(320)
        self.spinBox_img_size.setMaximum(1280)
        self.spinBox_img_size.setSingleStep(32)
        self.spinBox_img_size.setObjectName("spinBox_img_size")
        
        # 设备选择标签
        self.label_device = QtWidgets.QLabel(self.groupBox_config)
        self.label_device.setGeometry(QtCore.QRect(20, 110, 100, 20))
        self.label_device.setObjectName("label_device")
        
        # 设备选择下拉框
        self.comboBox_device = QtWidgets.QComboBox(self.groupBox_config)
        self.comboBox_device.setGeometry(QtCore.QRect(120, 110, 200, 20))
        self.comboBox_device.setObjectName("comboBox_device")
        
        # 验证间隔标签
        self.label_val_interval = QtWidgets.QLabel(self.groupBox_config)
        self.label_val_interval.setGeometry(QtCore.QRect(350, 110, 100, 20))
        self.label_val_interval.setObjectName("label_val_interval")
        
        # 验证间隔输入框
        self.spinBox_val_interval = QtWidgets.QSpinBox(self.groupBox_config)
        self.spinBox_val_interval.setGeometry(QtCore.QRect(450, 110, 100, 20))
        self.spinBox_val_interval.setMinimum(1)
        self.spinBox_val_interval.setMaximum(100)
        self.spinBox_val_interval.setObjectName("spinBox_val_interval")
        
        # 训练部分
        self.groupBox_train = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_train.setGeometry(QtCore.QRect(20, 420, 760, 120))
        self.groupBox_train.setObjectName("groupBox_train")
        
        # 训练按钮
        self.pushButton_train = QtWidgets.QPushButton(self.groupBox_train)
        self.pushButton_train.setGeometry(QtCore.QRect(100, 40, 200, 40))
        self.pushButton_train.setObjectName("pushButton_train")
        
        # 取消按钮
        self.pushButton_cancel = QtWidgets.QPushButton(self.groupBox_train)
        self.pushButton_cancel.setGeometry(QtCore.QRect(400, 40, 200, 40))
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        
        # 训练进度条
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(20, 560, 760, 20))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        
        # 状态标签
        self.label_status = QtWidgets.QLabel(self.centralwidget)
        self.label_status.setGeometry(QtCore.QRect(20, 540, 760, 20))
        self.label_status.setObjectName("label_status")
        
        TrainMainWindow.setCentralWidget(self.centralwidget)
        
        self.retranslateUi(TrainMainWindow)
        QtCore.QMetaObject.connectSlotsByName(TrainMainWindow)
    
    def retranslateUi(self, TrainMainWindow):
        _translate = QtCore.QCoreApplication.translate
        TrainMainWindow.setWindowTitle(_translate("TrainMainWindow", "模型训练"))
        self.label_title.setText(_translate("TrainMainWindow", "基于YOLOv8的模型训练"))
        self.groupBox_data.setTitle(_translate("TrainMainWindow", "数据集设置"))
        self.label_data_path.setText(_translate("TrainMainWindow", "数据集路径:"))
        self.pushButton_browse.setText(_translate("TrainMainWindow", "浏览"))
        self.pushButton_upload_images.setText(_translate("TrainMainWindow", "上传图片"))
        self.pushButton_upload_folder.setText(_translate("TrainMainWindow", "上传文件夹"))
        self.groupBox_config.setTitle(_translate("TrainMainWindow", "训练配置"))
        self.label_model_type.setText(_translate("TrainMainWindow", "模型类型:"))
        self.label_epochs.setText(_translate("TrainMainWindow", "训练轮数:"))
        self.label_batch.setText(_translate("TrainMainWindow", "批次大小:"))
        self.label_lr.setText(_translate("TrainMainWindow", "学习率:"))
        self.label_img_size.setText(_translate("TrainMainWindow", "图片大小:"))
        self.label_device.setText(_translate("TrainMainWindow", "设备选择:"))
        self.label_val_interval.setText(_translate("TrainMainWindow", "验证间隔:"))
        self.groupBox_train.setTitle(_translate("TrainMainWindow", "训练控制"))
        self.pushButton_train.setText(_translate("TrainMainWindow", "开始训练"))
        self.pushButton_cancel.setText(_translate("TrainMainWindow", "取消"))
        self.label_status.setText(_translate("TrainMainWindow", "状态: 就绪"))

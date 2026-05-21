# -*- coding: utf-8 -*-
"""
加载动画组件
"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QMovie

class LoadingAnimation(QWidget):
    """加载动画类"""
    
    def __init__(self, parent=None, text="加载中..."):
        """初始化加载动画
        
        Args:
            parent: 父窗口
            text: 加载提示文本
        """
        super(LoadingAnimation, self).__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 设置布局
        layout = QVBoxLayout(self)
        
        # 创建加载动画标签
        self.movie_label = QLabel(self)
        self.movie = QMovie("UIProgram/ui_imgs/icons/loading.gif")  # 假设我们有一个loading.gif动画
        self.movie_label.setMovie(self.movie)
        self.movie_label.setAlignment(Qt.AlignCenter)
        
        # 创建文本标签
        self.text_label = QLabel(text, self)
        self.text_label.setAlignment(Qt.AlignCenter)
        
        # 添加到布局
        layout.addWidget(self.movie_label)
        layout.addWidget(self.text_label)
        
        # 调整大小
        self.setFixedSize(200, 150)
    
    def start(self):
        """开始动画"""
        self.movie.start()
        self.show()
    
    def stop(self):
        """停止动画"""
        self.movie.stop()
        self.hide()
    
    def setText(self, text):
        """设置文本
        
        Args:
            text: 提示文本
        """
        self.text_label.setText(text)

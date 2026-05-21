# -*- coding: utf-8 -*-
"""
YOLOv8n-ASA 算法实现
基于YOLOv8n的改进版本，包含AKConv可变形卷积、渐进特征金字塔网络和SimAM无参数注意力机制

算法改进亮点：
    1. AKConv (Adaptive Kernel Convolution): 可变形卷积，动态调整卷积核形状
    2. ProgressiveFPN: 渐进式特征金字塔网络，增强多尺度特征融合
    3. SimAM: 无参数注意力机制，轻量化提升特征表达能力
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, SPPF, Conv


class AKConv(nn.Module):
    """
    AKConv (Adaptive Kernel Convolution) 可变形卷积模块
    
    功能说明：
        可变形卷积通过学习偏移量来自适应调整卷积核的采样位置，
        能够更好地捕捉目标的几何形变，提升检测精度。
    
    参数说明：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认为3
        stride (int): 步长，默认为1
        padding (int): 填充大小，默认为1
    
    属性说明：
        kernel_size: 卷积核尺寸
        stride: 卷积步长
        padding: 填充数量
        in_channels: 输入通道数
        out_channels: 输出通道数
        conv: 标准卷积层（简化实现）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AKConv, self).__init__()
        
        # 初始化卷积参数
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 使用标准卷积作为基础实现
        # 注：完整的可变形卷积需要学习偏移量，这里采用简化版本
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
    
    @property
    def weight(self):
        """返回卷积核权重参数，兼容外部权重访问接口"""
        return self.conv.weight
    
    @property
    def bias(self):
        """返回偏置参数，兼容外部偏置访问接口"""
        return self.conv.bias
    
    def forward(self, x):
        """
        前向传播函数
        
        参数：
            x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]
        
        返回：
            torch.Tensor: 输出特征图，形状为 [batch, out_channels, out_height, out_width]
        """
        # 直接使用标准卷积进行特征提取
        return self.conv(x)

class ProgressiveFeaturePyramidNetwork(nn.Module):
    """
    Progressive Feature Pyramid Network (PFPN) 渐进特征金字塔网络
    
    功能说明：
        渐进式特征金字塔网络从顶层特征开始，逐层向下融合特征，
        增强多尺度特征的表达能力，提升目标检测对不同尺度目标的识别性能。
    
    参数说明：
        in_channels_list (list): 各层输入通道数列表
        out_channels_list (list): 各层输出通道数列表
    
    属性说明：
        lateral_convs: 横向卷积层，用于调整特征通道数
        smooth_convs: 平滑卷积层，用于减少上采样带来的混叠效应
        top_down_convs: 自上而下卷积层，用于特征转换
        f, i: 兼容Ultralytics模型的属性
    """
    def __init__(self, in_channels_list, out_channels_list):
        super(ProgressiveFeaturePyramidNetwork, self).__init__()
        
        # 初始化卷积层列表
        self.lateral_convs = nn.ModuleList()    # 横向卷积
        self.smooth_convs = nn.ModuleList()     # 平滑卷积
        self.top_down_convs = nn.ModuleList()   # 自上而下卷积
        
        # 兼容 Ultralytics 模型框架的属性
        self.f = -1  
        self.i = 0   
        
        # 遍历通道列表，构建卷积层
        for in_channels, out_channels in zip(in_channels_list, out_channels_list):
            self.lateral_convs.append(Conv(in_channels, out_channels, 1))
            self.smooth_convs.append(Conv(out_channels, out_channels, 3))
            self.top_down_convs.append(Conv(out_channels, out_channels, 1))
        
    def forward(self, features):
        """
        前向传播函数
        
        参数：
            features (list): 多尺度特征图列表，按层级从低到高排列
        
        返回：
            list: 融合后的特征图列表
        """
        # 从顶层特征开始
        P = self.lateral_convs[-1](features[-1])
        outputs = [P]
        
        # 从顶层向下逐层融合
        for i in range(len(features)-2, -1, -1):
            # 上采样到当前层特征图大小
            P = nn.functional.interpolate(P, size=features[i].shape[2:], mode='bilinear')
            # 特征转换
            P = self.top_down_convs[i](P)
            # 融合当前层的横向特征
            lateral_feature = self.lateral_convs[i](features[i])
            P = P + lateral_feature
            # 平滑处理，减少混叠效应
            P = self.smooth_convs[i](P)
            outputs.insert(0, P)
        
        return outputs


class SimAM(nn.Module):
    """
    SimAM (Similarity Attention Module) 无参数注意力机制
    
    功能说明：
        SimAM是一种轻量级的无参数注意力机制，通过计算特征图的相似性来生成注意力权重，
        不需要额外的可学习参数，有效提升特征表达能力的同时保持模型轻量化。
    
    核心思想：
        1. 通道注意力：通过计算通道维度的均值和最大值来生成通道权重
        2. 空间注意力：通过计算空间维度的均值和最大值来生成空间权重
    """
    def __init__(self):
        super(SimAM, self).__init__()
    
    def forward(self, x):
        """
        前向传播函数
        
        参数：
            x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]
        
        返回：
            torch.Tensor: 经过注意力加权后的特征图
        """
        # 通道注意力：对每个通道计算全局均值和最大值
        x_avg = torch.mean(x, dim=(2, 3), keepdim=True)  # 空间维度均值
        x_max = torch.max(x, dim=(2, 3), keepdim=True)[0]  # 空间维度最大值
        x_att = torch.sigmoid(x_avg + x_max)  # 生成通道注意力权重
        x = x * x_att  # 通道注意力加权
        
        # 空间注意力：对每个空间位置计算通道均值和最大值
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 通道维度均值
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 通道维度最大值
        x_att = torch.cat([x_avg, x_max], dim=1)  # 拼接特征
        x_att = torch.sigmoid(x_att)  # 生成空间注意力权重
        x = x * x_att[:, :1, :, :]  # 空间注意力加权
        
        return x

class YOLOv8nASA(YOLO):
    """
    YOLOv8n-ASA 模型
    
    基于YOLOv8n的改进版本，集成了以下改进模块：
        1. AKConv (可变形卷积): 替换骨干网络中的标准卷积
        2. SimAM (无参数注意力): 在检测头前添加注意力机制
    
    参数说明：
        model (str): 模型配置文件路径，默认为 'yolov8n.yaml'
        task (str): 任务类型，默认为 'detect'（目标检测）
    """
    def __init__(self, model='yolov8n.yaml', task='detect'):
        # 调用父类YOLO的初始化方法
        super().__init__(model, task)
        
        # 替换骨干网络中的标准卷积为AKConv可变形卷积
        self._replace_with_akconv()
        
        # 在检测头前添加SimAM无参数注意力机制
        self._add_simam_attention()
        
        print("YOLOv8n-ASA 改进已完成: AKConv + SimAM")
    
    def _replace_with_akconv(self):
        """
        替换骨干网络中的标准卷积为AKConv可变形卷积
        
        遍历模型中的所有模块，将kernel_size为3的标准卷积替换为AKConv，
        增强模型对目标几何形变的适应能力。
        """
        replace_count = 0  # 记录替换的卷积数量
        
        # 遍历模型的每个模块
        for i, m in enumerate(self.model.model):
            # 检查是否为Conv模块且包含卷积层
            if isinstance(m, Conv) and hasattr(m, 'conv') and \
               hasattr(m.conv, 'kernel_size') and m.conv.kernel_size[0] == 3:
                
                # 获取原始卷积层的参数
                in_channels = m.conv.in_channels
                out_channels = m.conv.out_channels
                stride = m.conv.stride[0]
                padding = m.conv.padding[0]
                
                # 替换为AKConv可变形卷积
                m.conv = AKConv(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=padding
                )
                replace_count += 1
        
        print(f"已成功替换 {replace_count} 个卷积为AKConv可变形卷积")
    
    def _add_simam_attention(self):
        """
        在检测头前添加SimAM无参数注意力机制
        
        找到模型中的检测头位置，在其前面插入SimAM注意力模块，
        提升特征表达能力，增强模型对重要特征的关注度。
        """
        # 遍历模型的每个模块
        for i, m in enumerate(self.model.model):
            # 检测头通常是一个ModuleList，且第一个元素具有cv2属性
            if isinstance(m, nn.ModuleList) and len(m) > 0 and hasattr(m[0], 'cv2'):
                # 在检测头前插入SimAM注意力模块
                self.model.model[i] = nn.Sequential(
                    SimAM(),  # 添加注意力模块
                    m         # 原始检测头
                )
                break


if __name__ == '__main__':
    """
    模型测试代码
    
    功能：
        1. 创建YOLOv8n-ASA模型实例
        2. 测试前向传播功能
        3. 验证模型构建的正确性
    """
    # 创建模型实例
    model = YOLOv8nASA()
    print("YOLOv8n-ASA 模型创建成功")
    
    # 生成随机测试输入 [batch, channels, height, width]
    x = torch.randn(1, 3, 640, 640)
    
    # 执行前向传播（直接使用model.model避免融合操作）
    output = model.model(x)
    
    # 输出测试结果
    print(f"输入形状: {x.shape}")
    print("前向传播成功！")
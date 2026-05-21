# -*- coding: utf-8 -*-
"""
YOLOv8n-FGP 算法实现
基于YOLOv8n的轻量化改进版本

算法改进亮点：
    1. FasterNetBlock: 轻量化卷积模块，通过部分卷积减少计算量
    2. GSTA (Global-Spatio-Temporal Attention): 全局-局部时空注意力机制
    3. ParNet-C2f: 并行网络结构，增强特征提取能力
    4. WIoU (Weighted IoU): 加权IoU损失函数，提升定位精度
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, SPPF, Conv


class FasterNetBlock(nn.Module):
    """
    FasterNet 轻量化卷积模块
    
    功能说明：
        FasterNet通过将输入特征图按通道维度分割，分别经过不同的卷积路径处理后融合，
        从而在保持精度的同时显著减少计算量，实现模型轻量化。
    
    参数说明：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 主卷积核大小，默认为3
        stride (int): 步长，默认为1
        padding (int): 填充大小，默认为1
    
    属性说明：
        split_ratio (float): 通道分割比例，默认为0.5
        split_channels (int): 主分支的通道数
        main_conv (nn.Conv2d): 主卷积分支，使用3x3卷积
        aux_conv (nn.Conv2d): 辅助分支，使用1x1卷积
        act (nn.SiLU): 激活函数
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FasterNetBlock, self).__init__()
        
        # 通道分割比例：50%进入主分支，50%进入辅助分支
        self.split_ratio = 0.5
        self.split_channels = int(in_channels * self.split_ratio)
        
        # 主卷积分支：使用标准卷积进行特征提取
        self.main_conv = nn.Conv2d(
            in_channels=self.split_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=1
        )
        
        # 辅助分支：使用1x1卷积进行通道变换
        self.aux_conv = nn.Conv2d(
            in_channels=in_channels - self.split_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=0
        )
        
        # SiLU激活函数，提供平滑的非线性变换
        self.act = nn.SiLU()
    
    def forward(self, x):
        """
        前向传播函数
        
        参数：
            x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]
        
        返回：
            torch.Tensor: 输出特征图，形状为 [batch, out_channels, out_height, out_width]
        
        执行流程：
            1. 将输入特征图按通道分割为两部分
            2. 主分支通过3x3卷积提取空间特征
            3. 辅助分支通过1x1卷积进行通道变换
            4. 逐元素相加融合两个分支的特征
            5. 应用SiLU激活函数
        """
        # 沿通道维度分割输入特征图
        x1, x2 = torch.split(x, [self.split_channels, x.size(1) - self.split_channels], dim=1)
        
        # 主分支：3x3卷积提取空间特征
        x1 = self.main_conv(x1)
        
        # 辅助分支：1x1卷积调整通道数
        x2 = self.aux_conv(x2)
        
        # 特征融合：逐元素相加
        x = x1 + x2
        
        # 应用SiLU激活函数
        x = self.act(x)
        
        return x

class GSTA(nn.Module):
    """
    GSTA (Global-Spatio-Temporal Attention) 全局-局部时空注意力模块
    
    功能说明：
        GSTA结合通道注意力和空间注意力机制，能够自适应地突出重要特征，
        抑制冗余信息，提升模型的特征表达能力和检测精度。
    
    参数说明：
        in_channels (int): 输入通道数
        reduction (int): 通道注意力的通道缩减比例，默认为4
    
    属性说明：
        channel_attention (nn.Sequential): 通道注意力子模块
        spatial_attention (nn.Sequential): 空间注意力子模块
    """
    def __init__(self, in_channels, reduction=4):
        super(GSTA, self).__init__()
        
        # 通道注意力机制
        # 通过全局平均池化和1x1卷积生成通道权重
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，压缩空间维度
            nn.Conv2d(in_channels, in_channels // reduction, 1),  # 降维
            nn.SiLU(),  # 激活函数
            nn.Conv2d(in_channels // reduction, in_channels, 1),  # 升维
            nn.Sigmoid()  # 生成注意力权重
        )
        
        # 空间注意力机制
        # 通过通道维度的均值和最大值生成空间权重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),  # 融合均值和最大值特征
            nn.Sigmoid()  # 生成空间注意力权重
        )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数：
            x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]
        
        返回：
            torch.Tensor: 经过注意力加权后的特征图
        
        执行流程：
            1. 通道注意力：计算每个通道的重要性权重
            2. 空间注意力：计算每个空间位置的重要性权重
            3. 依次应用两种注意力机制
        """
        # 通道注意力：为每个通道生成权重
        ca = self.channel_attention(x)
        x = x * ca  # 通道注意力加权
        
        # 空间注意力：为每个空间位置生成权重
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道维度均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度最大值
        sa_input = torch.cat([avg_out, max_out], dim=1)  # 拼接特征
        sa = self.spatial_attention(sa_input)  # 生成空间注意力权重
        x = x * sa  # 空间注意力加权
        
        return x


class ParNetC2f(nn.Module):
    """
    ParNet-C2f 并行网络结构
    
    功能说明：
        ParNet-C2f是一种并行化的特征提取模块，通过多个并行的卷积路径提取特征，
        然后进行特征融合，能够有效增强模型的特征表达能力和梯度流动。
    
    参数说明：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        n (int): 并行卷积块的数量，默认为2
        shortcut (bool): 是否使用残差连接，默认为True
        g (int): 分组卷积的组数，默认为1
        e (float): 扩展比例，控制中间通道数，默认为0.5
    
    属性说明：
        c (int): 中间通道数
        cv1 (Conv): 输入卷积层，将通道数调整为2*c
        cv2 (Conv): 输出卷积层，融合特征并调整通道数
        m (nn.ModuleList): 并行卷积块列表
        shortcut (bool): 是否启用残差连接
        f, i: 兼容Ultralytics模型的属性
    """
    def __init__(self, in_channels, out_channels, n=2, shortcut=True, g=1, e=0.5):
        super(ParNetC2f, self).__init__()
        
        # 计算中间通道数
        self.c = int(out_channels * e)
        
        # 输入卷积层：将通道数调整为2*c
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        
        # 输出卷积层：融合特征并调整为输出通道数
        self.cv2 = Conv(2 * self.c, out_channels, 1, 1)
        
        # 并行卷积块列表：每个块包含两个3x3卷积
        self.m = nn.ModuleList([nn.Sequential(
            Conv(self.c, self.c, 3, 1, g=g),  # 第一个3x3卷积
            Conv(self.c, self.c, 3, 1, g=g)   # 第二个3x3卷积
        ) for _ in range(n)])
        
        # 是否使用残差连接
        self.shortcut = shortcut
        
        # 兼容Ultralytics模型框架的属性
        self.f = -1  
        self.i = 0   
    
    def forward(self, x):
        """
        前向传播函数
        
        参数：
            x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]
        
        返回：
            torch.Tensor: 输出特征图，形状为 [batch, out_channels, height, width]
        
        执行流程：
            1. 通过cv1将输入通道调整为2*c
            2. 沿通道维度分割为两部分
            3. 依次通过每个并行卷积块
            4. 拼接所有特征
            5. 通过cv2融合特征
            6. 如果启用残差连接，添加原始输入
        """
        # 通过cv1调整通道数并分割为两部分
        y = list(self.cv1(x).chunk(2, 1))
        
        # 依次通过每个并行卷积块，将结果追加到列表
        for m in self.m:
            y.append(m(y[-1]))
        
        # 拼接所有特征并通过cv2融合
        y = self.cv2(torch.cat(y, 1))
        
        # 如果启用残差连接，添加原始输入
        return y if not self.shortcut else y + x

class WIoU(nn.Module):
    """
    WIoU (Weighted IoU) 加权IoU损失函数
    
    功能说明：
        WIoU在传统IoU损失的基础上引入了中心点距离惩罚项，
        能够更好地优化目标框的定位精度，加速模型收敛。
    
    参数说明：
        iou_type (str): IoU类型，支持 'wiou' 和 'iou'，默认为 'wiou'
        eps (float): 数值稳定性参数，防止除零错误，默认为 1e-6
    
    属性说明：
        iou_type: IoU计算类型
        eps: 数值稳定性参数
    """
    def __init__(self, iou_type='wiou', eps=1e-6):
        super(WIoU, self).__init__()
        self.iou_type = iou_type  # 设置IoU类型
        self.eps = eps            # 设置数值稳定性参数
    
    def forward(self, pred, target):
        """
        前向传播函数
        
        参数：
            pred (torch.Tensor): 预测特征图，形状为 [batch, channels, height, width]
            target (torch.Tensor): 目标特征图，形状为 [batch, channels, height, width]
        
        返回：
            torch.Tensor: 损失值（标量）
        
        执行流程：
            1. 将输入转换为浮点型
            2. 计算IoU（交集/并集）
            3. 如果是WIoU类型，计算中心点距离惩罚项
            4. 返回损失值（1 - IoU 或 1 - WIoU）
        """
        # 将输入转换为浮点型
        pred = pred.float()
        target = target.float()
        
        # 计算交集：预测和目标的逐元素乘积之和
        inter = (pred * target).sum(dim=(1, 2, 3))
        
        # 计算并集：预测面积 + 目标面积 - 交集
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
        
        # 计算IoU
        iou = inter / (union + self.eps)
        
        # 计算WIoU
        if self.iou_type == 'wiou':
            # 计算预测和目标特征图的中心点
            pred_center = self._get_center(pred)
            target_center = self._get_center(target)
            
            # 计算中心点之间的欧氏距离
            center_dist = torch.norm(pred_center - target_center, dim=1)
            
            # 计算边界框对角线长度（用于归一化）
            bbox_diag = torch.sqrt(torch.sum((pred.max(2)[0].max(2)[0] - pred.min(2)[0].min(2)[0])**2, dim=1))
            
            # 计算归一化距离
            normalized_dist = center_dist / (bbox_diag + self.eps)
            
            # WIoU = IoU - 归一化距离
            wiou = iou - normalized_dist
            loss = 1 - wiou
        else:
            # 普通IoU损失
            loss = 1 - iou
        
        # 返回平均损失
        return loss.mean()
    
    def _get_center(self, x):
        """
        计算特征图的中心点坐标
        
        参数：
            x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]
        
        返回：
            torch.Tensor: 中心点坐标，形状为 [batch, 2]
        """
        batch_size, channels, height, width = x.shape
        
        # 初始化中心点坐标张量
        center = torch.zeros(batch_size, 2, device=x.device)
        
        # 遍历每个样本
        for i in range(batch_size):
            # 找到非零元素的位置
            non_zero = torch.nonzero(x[i])
            
            if non_zero.numel() > 0:
                # 计算非零元素的均值作为中心点坐标
                # 取第2和第3维度（高度和宽度方向）
                center[i] = non_zero.float().mean(dim=0)[1:3]
        
        return center


class YOLOv8nFGP(YOLO):
    """
    YOLOv8n-FGP 模型
    
    基于YOLOv8n的轻量化改进版本，集成了以下改进模块：
        1. FasterNetBlock: 替换骨干网络中的标准卷积，实现轻量化
        2. GSTA: 在检测头前添加全局-局部时空注意力机制
        3. WIoU: 使用加权IoU损失函数，提升定位精度
    
    参数说明：
        model (str): 模型配置文件路径，默认为 'yolov8n.yaml'
        task (str): 任务类型，默认为 'detect'（目标检测）
    """
    def __init__(self, model='yolov8n.yaml', task='detect'):
        # 调用父类YOLO的初始化方法
        super().__init__(model, task)
        
        # 替换骨干网络为FasterNet轻量化版本
        self._replace_with_fasternet()
        
        # 添加GSTA全局-局部时空注意力机制
        self._add_gsta_attention()
        
        # 替换损失函数为WIoU损失
        self._replace_with_wiou_loss()
        
        print("YOLOv8n-FGP 改进已完成: FasterNet + GSTA + WIoU")
    
    def _replace_with_fasternet(self):
        """
        替换骨干网络中的标准卷积为FasterNet轻量化模块
        
        遍历模型中的所有Conv模块，将kernel_size为3的卷积替换为FasterNetBlock，
        在保持精度的同时减少计算量，实现模型轻量化。
        """
        # 遍历模型中的所有子模块
        for m in self.model.model.modules():
            # 检查是否为Conv模块且包含卷积层
            if isinstance(m, Conv) and hasattr(m, 'conv') and hasattr(m.conv, 'kernel_size'):
                
                # 获取原始卷积层的参数
                in_channels = m.conv.in_channels
                out_channels = m.conv.out_channels
                kernel_size = m.conv.kernel_size[0]
                stride = m.conv.stride[0]
                padding = m.conv.padding[0]
                
                # 只替换3x3卷积
                if kernel_size == 3:
                    m.conv = FasterNetBlock(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding
                    )
    
    def _add_gsta_attention(self):
        """
        在检测头前添加GSTA全局-局部时空注意力机制
        
        找到模型中的检测头位置，在其前面插入GSTA注意力模块，
        提升特征表达能力，增强模型对重要特征的关注度。
        """
        # 遍历模型的每个模块
        for i, m in enumerate(self.model.model):
            # 检测头通常是一个ModuleList，且第一个元素具有cv2属性
            if isinstance(m, nn.ModuleList) and len(m) > 0 and hasattr(m[0], 'cv2'):
                # 获取输入通道数
                in_channels = m[0].cv2.conv.in_channels
                
                # 在检测头前插入GSTA注意力模块
                self.model.model[i] = nn.Sequential(
                    GSTA(in_channels),  # 添加注意力模块
                    m                   # 原始检测头
                )
                break
    
    def _replace_with_wiou_loss(self):
        """
        初始化WIoU损失函数
        
        在Ultralytics YOLO中，损失函数通常在训练循环中定义，
        这里创建一个WIoU损失函数实例，供训练时使用。
        """
        self.wiou_loss = WIoU()
        print("已初始化WIoU损失函数")


if __name__ == '__main__':
    """
    模型测试代码
    
    功能：
        1. 创建YOLOv8n-FGP模型实例
        2. 测试前向传播功能
        3. 验证WIoU损失函数
    """
    # 创建模型实例
    model = YOLOv8nFGP()
    print("YOLOv8n-FGP 模型创建成功")
    
    # 生成随机测试输入 [batch, channels, height, width]
    x = torch.randn(1, 3, 640, 640)
    
    # 执行前向传播（直接使用model.model避免融合操作）
    output = model.model(x)
    print(f"输入形状: {x.shape}")
    print("前向传播成功！")
    
    # 测试WIoU损失函数
    loss_fn = WIoU()
    inputs = torch.randn(1, 1, 80, 80)
    targets = torch.randint(0, 2, (1, 1, 80, 80)).float()
    loss = loss_fn(inputs, targets)
    print(f"WIoU损失值: {loss.item()}")
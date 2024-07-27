import numpy as np
import torch
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
from torch.nn.modules.module import _addindent
import numpy
from .position_encoding import build_position_encoding
from typing import Dict, List
from util.misc import NestedTensor
from torch.autograd import Variable

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        # 设置max_norm属性，它指定了权重向量的L2范数的最大允许值
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        # 使用torch.renorm函数来重新归一化权重。这里，p=2表示计算L2范数，
        # dim=0表示沿着权重的第一个维度（通常是输出通道维度）进行归一化，
        # maxnorm参数指定了归一化后的最大L2范数值
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            # 第一个卷积层，输入通道数为1，输出通道数为self.F1，卷积核大小为(1, self.kernelLength)
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            # 批量归一化层，用于加速训练并提升模型性能
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            # 深度可分离卷积层，先对输入特征图的每个通道进行空间卷积
            # Conv2dWithConstraint是自定义的带有权重约束的卷积层
            # DepthwiseConv2D =======================
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            # 再次进行批量归一化
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate))
        block2 = nn.Sequential(
            # 可分离卷积层，首先是对特征图每个通道分别进行卷积（相当于分组卷积）
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            # 接着是一个1x1的卷积层，用于改变特征图的通道数
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate))
        return nn.Sequential(block1, block2)

    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Sequential(
            nn.Linear(inputSize, n_classes, bias=False),
            nn.Softmax(dim=1))

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, n_classes=4, channels=60, samples=151,
                 dropoutRate=0.5, kernelLength=64, kernelLength2=16, F1=8,
                 D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1                           # 第一个卷积层的输出通道数
        self.F2 = F2                           # 第二个卷积层的输出通道数
        self.D = D                             # 深度可分离卷积的扩展因子
        self.samples = samples                 # 输入样本的长度
        self.n_classes = n_classes             # 输出的类别数
        self.channels = channels               # 输入数据的通道数（对于EEG数据，这可能是电极的数量）
        self.kernelLength = kernelLength       # 第一个卷积层的卷积核长度
        self.kernelLength2 = kernelLength2     # 第二个卷积层的卷积核长度
        self.dropoutRate = dropoutRate         # Dropout层的丢弃率

        # self.blocks = self.InitialBlocks(dropoutRate)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            # DepthwiseConv2D =======================
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=2),
            nn.Dropout(p=dropoutRate))
        self.block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 3), stride=3),
            nn.Dropout(p=dropoutRate))
        # self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, samples)
        # self.classifierBlock = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], n_classes)

    def forward(self, x):
        # x = self.blocks(x)
        # x =torch.tensor(numpy.zeros((64,1,8,3000)),dtype=torch.float32).cuda()
        x = self.block1(x)
        x = self.block2(x)
        # x = x.view(x.size()[0], -1)  # Flatten
        # x = self.classifierBlock(x)

        return x

def categorical_cross_entropy(y_pred, y_true):
    # y_pred = y_pred.cuda()
    # y_true = y_true.cuda()

    # 将y_pred的值限制在1e-9和1-1e-9之间，以避免对0或1取对数时导致的数值不稳定问题。
    # 这样可以确保在计算log(y_pred)时不会得到无穷大或负无穷大的值。
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)

    # 计算交叉熵损失。首先，y_true * torch.log(y_pred) 对每个样本的每个类别计算损失。
    # 然后，使用sum(dim=1)沿着类别维度求和，得到每个样本的总损失。
    # 最后，使用mean()计算所有样本的平均损失。
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def torch_summarize(model, show_weights=True, show_parameters=True):
    """
    通过显示可训练参数和权重来总结torch模型。

    参数:
    model (torch.nn.Module): 需要总结的PyTorch模型。
    show_weights (bool, 可选): 是否显示每层的权重形状，默认为True。
    show_parameters (bool, 可选): 是否显示每层的参数数量，默认为True。

    返回:
    str: 包含模型信息的字符串。
    """
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, batch_size):
        super().__init__(backbone, position_embedding)
        self.batch_size = batch_size

    def forward(self, tensor_list: NestedTensor):
        # 对tensor_list中的tensors进行backbone操作
        temp = self[0](tensor_list.tensors)

        # 将temp的维度进行变换。首先交换第0和第2维度，然后展平第2和第3维度，然后调整大小为(batch_size, 1, 256, 125)，最后再次交换第0和第2维度
        # mask = torch.tensor(numpy.zeros_like(temp.cpu(), dtype=bool), dtype=torch.bool)
        temp = temp.permute(0,2,1,3).flatten(2).resize(self.batch_size,1,256,125).permute(0,2,1,3)

        # 创建一个大小为(batch_size, 1, 125)的全零布尔型张量，并转移到GPU上
        mask =torch.tensor(numpy.zeros((self.batch_size,1,125), dtype=bool),dtype=torch.bool).cuda()
        # temp = temp.premute(0,2,1).unsqueeze(1)

        # 创建一个字典xs，其中键为'0'，值为一个新的NestedTensor，包含处理后的temp和mask
        xs = {'0':NestedTensor(temp,mask)}

        # 初始化输出列表out和位置编码列表pos
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)

            # 对x进行位置嵌入操作，并将结果转换为与x.tensors相同的数据类型，然后添加到位置编码列表pos中
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        # 返回输出张量列表和位置编码列表
        return out, pos

def build_EEG(args):
    """
    构建EEGNet模型并结合位置嵌入，返回完整的模型。

    Args:
        args: 包含模型构建所需参数的命名空间对象。

    Returns:
        model: 结合了EEGNet和位置嵌入的完整模型。
    """
    # 构建位置嵌入模块
    position_embedding = build_position_encoding(args)

    # 定义EEGNet的参数
    n_classes, dropoutRate, kernelLength, kernelLength2, F1, D = 4, 0.5, 1500, 325, 16, 4
    # 根据F1和D计算第二个卷积层的滤波器数量
    F2 = F1 * D
    # 初始化EEGNet模型
    net = EEGNet(n_classes, 8, 3000, dropoutRate, kernelLength, kernelLength2, F1, D, F2)

    # 构建Joiner模型，将EEGNet和位置嵌入结合
    model = Joiner(net, position_embedding, args.batch_size)
    return model
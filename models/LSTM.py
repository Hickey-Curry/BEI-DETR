import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(1)
#
# lstm = nn.LSTM(3, 3)  # 输入单词用一个维度为3的向量表示, 隐藏层的一个维度3，仅有一层的神经元，
# # 记住就是神经元，这个时候神经层的详细结构还没确定，仅仅是说这个网络可以接受[seq_len,batch_size,3]的数据输入
# print(lstm.all_weights)
#
# inputs = [torch.randn(1, 3) for _ in range(5)]
# # 构造一个由5个单单词组成的句子 构造出来的形状是 [5,1,3]也就是明确告诉网络结构我一个句子由5个单词组成，
# # 每个单词由一个1X3的向量组成，就是这个样子[1,2,3]
# # 同时确定了网络结构，每个批次只输入一个句子，其中第二维的batch_size很容易迷惑人
# # 对整个这层来说，是一个批次输入多少个句子，具体但每个神经元，就是一次性喂给神经元多少个单词。
# print('Inputs:', inputs)
#
# # 初始化隐藏状态
# hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
# print('Hidden:', hidden)
# for i in inputs:
#     # 将序列的元素逐个输入到LSTM，这里的View是把输入放到第三维，看起来有点古怪，
#     # 回头看看上面的关于LSTM输入的描述，这是固定的格式，以后无论你什么形式的数据，
#     # 都必须放到这个维度。就是在原Tensor的基础之上增加一个序列维和MiniBatch维，
#     # 这里可能还会有迷惑，前面的1是什么意思啊，就是一次把这个输入处理完，
#     # 在输入的过程中不会输出中间结果，这里注意输入的数据的形状一定要和LSTM定义的输入形状一致。
#     # 经过每步操作,hidden 的值包含了隐藏状态的信息
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
# print('out1:', out)
# print('hidden2:', hidden)
# # 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# # 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# # 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# # 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.
#
# # 增加额外的第二个维度
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print('out2', out)
# print('hidden3', hidden)
# #
from .position_encoding import build_position_encoding
from typing import Dict, List
from util.misc import NestedTensor
from torch.autograd import Variable
import numpy

class LSTM(nn.Module):
    def __init__(self, inpust_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(inpust_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x= x
        out = self.fc(x)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        temp = self[0](tensor_list.tensors.permute(0,2,1))
        # mask = torch.tensor(numpy.zeros_like(temp.cpu(), dtype=bool), dtype=torch.bool)
        mask =torch.tensor(numpy.zeros(temp.shape, dtype=bool),dtype=torch.bool).cuda()
        temp = temp.premute(0,2,1).unsqueeze(1)
        xs = {'0':NestedTensor(temp,mask)}
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))


        return out, pos

def build_LSTM_EYE(args):
    position_embedding = build_position_encoding(args)
    lstm = LSTM(inpust_dim=8, hidden_dim=12, layer_dim=6, output_dim=8)
    model = Joiner(lstm, position_embedding)
    return model
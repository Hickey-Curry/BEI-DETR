# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_size=1 ,normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        """
        d_model: 编码器里面mlp（前馈神经网络  2个linear层）的hidden dim 512
        nhead: 多头注意力头数 8
        num_encoder_layers: encoder的层数 6
        num_decoder_layers: decoder的层数 6
        dim_feedforward: 前馈神经网络的维度 2048
        dropout: 0.1
        activation: 激活函数类型 relu
        normalize_before: 是否使用前置LN
        return_intermediate_dec: 是否返回decoder中间层结果  False
        """
        # 初始化一个小encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        # 创建整个Encoder层  6个encoder层堆叠
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.fusion_encoder1 = FousionModelEncoder(encoder_layer, num_encoder_layers, encoder_norm, batch_size)

        encoder_layer1 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm1 = nn.LayerNorm(d_model) if normalize_before else None
        self.fusion_encoder2 = FousionModelEncoder(encoder_layer1, num_encoder_layers, encoder_norm1,batch_size)

        # 初始化一个小decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        # 创建整个Decoder层  6个decoder层堆叠
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 参数初始化
        self._reset_parameters()

        self.d_model = d_model    # 编码器里面mlp的hidden dim 512
        self.nhead = nhead        # 多头注意力头数 8

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src1, src2, src3, mask1, query_embed1, pos_embed1, mask2, query_embed2, pos_embed2, mask3, query_embed3, pos_embed3):
        """
        src: [bs,256,19,26] 图片输入backbone+1x1conv之后的特征图
        mask: [bs, 19, 26]  用于记录特征图中哪些地方是填充的（原图部分值为False，填充部分值为True）
        query_embed: [100, 256]  类似于传统目标检测里面的anchor  这里设置了100个   需要预测的目标
        pos_embed: [bs, 256, 19, 26]  位置编码
        """
        # bs  c=256  h=19  w=26
        # flatten NxCxHxW to HWxNxC
        bs1, c1, h1, w1 = src1.shape
        # src: [bs,256,19,26]=[bs,C,H,W] -> [494,bs,256]=[HW,bs,C]
        src1 = src1.flatten(2).permute(2, 0, 1)
        bs2, c2, h2, w2 = src2.shape
        src2 = src2.flatten(2).permute(2, 0, 1)
        bs3, c3, h3, w3 = src3.shape
        src3 = src3.flatten(2).permute(2, 0, 1)

        # pos_embed: [bs, 256, 19, 26]=[bs,C,H,W] -> [494,bs,256]=[HW,bs,C]
        pos_embed1 = pos_embed1.flatten(2).permute(2, 0, 1)
        # pos_embed1 = pos_embed1.permute(2, 0, 1)

        # query_embed: [100, 256]=[num,C] -> [100,bs,256]=[num,bs,256]
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs1, 1)
        # pos_embed2 = pos_embed2.flatten(2).permute(2, 0, 1)
        pos_embed2 = pos_embed2.flatten(2).permute(2, 0, 1)
        query_embed2 = query_embed2.unsqueeze(1).repeat(1, bs1, 1)
        pos_embed3 = pos_embed3.flatten(2).permute(2, 0, 1)
        query_embed3 = query_embed3.unsqueeze(1).repeat(1, bs1, 1)

        # mask: [bs, 19, 26]=[bs,H,W] -> [bs,494]=[bs,HW]
        mask1 = mask1.flatten(1)
        mask2 = mask2.flatten(1)
        mask3 = mask3.flatten(1)

        # tgt: [100, bs, 256] 需要预测的目标query embedding 和 query_embed形状相同  且全设置为0
        #                     在每层decoder层中不断的被refine，相当于一次次的被coarse-to-fine的过程
        tgt1 = torch.zeros_like(query_embed1)
        tgt2 = torch.zeros_like(query_embed2)

        # memory: [494, bs, 256]=[HW, bs, 256]  Encoder输出  具有全局相关性（增强后）的特征表示
        # memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        memory1, pos12 = self.fusion_encoder1(src2, src1, src_key_padding_mask=mask1, pos1=pos_embed2, pos2=pos_embed1)
        memory2, pos23 = self.fusion_encoder2(src3, src1, src_key_padding_mask=mask1, pos1=pos_embed3, pos2=pos_embed1)

        # [6, 100, bs, 256]
        # tgt：需要预测的目标 query embeding
        # memory: encoder的输出
        # pos: memory的位置编码
        # query_pos: tgt的位置编码
        hs1 = self.decoder(tgt1, memory1, memory_key_padding_mask=mask1,
                          pos=pos12, query_pos=query_embed1)
        hs2 = self.decoder(tgt2, memory2, memory_key_padding_mask=mask1,
                          pos=pos23, query_pos=query_embed2)

        # decoder输出 [6, 100, bs, 256] -> [6, bs, 100, 256]
        # encoder输出 [bs, 256, H, W]
        return hs1.transpose(1, 2), hs2.transpose(1,2) #, memory1.permute(1, 2, 0).view(bs1, c1, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 复制num_layers=6份encoder_layer=TransformerEncoderLayer
        self.layers = _get_clones(encoder_layer, num_layers)
        # 6层TransformerEncoderLayer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        src: [h*w, bs, 256]  经过Backbone输出的特征图（降维到256）
        mask: None
        src_key_padding_mask: [h*w, bs]  记录每个特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256] 每个特征图的位置编码
        """
        output = src

        # 遍历这6层TransformerEncoderLayer
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        # 得到最终ENCODER的输出 [h*w, bs, 256]
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 复制num_layers=decoder_layer=TransformerDecoderLayer
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers  # 6
        self.norm = norm              # LN
        # 是否返回中间层 默认True  因为DETR默认6个Decoder都会返回结果，一起加入损失计算的
        # 每一层Decoder都是逐层解析，逐层加强的，所以前面层的解析效果对后面层的解析是有意义的，所以作者把前面5层的输出也加入损失计算
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        """
        tgt: [100, bs, 256] 需要预测的目标query embedding 和 query_embed形状相同  且全设置为0
                            在每层decoder层中不断的被refine，相当于一次次的被coarse-to-fine的过程
        memory: [h*w, bs, 256]  Encoder输出  具有全局相关性（增强后）的特征表示
        tgt_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]                 特征图的位置编码
        query_pos: [100, bs, 256]    query embedding的位置编码  随机初始化的
        """
        output = tgt   # 初始化query embedding  全是0

        intermediate = []   # 用于存放6层decoder的输出结果

        # 遍历6层decoder
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # 6层结果全部加入intermediate
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # 默认执行这里
        # 最后把  6x[100,bs,256] -> [6(6层decoder输出),100,bs,256]
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)  # 不执行


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        """
        小encoder层  结构：multi-head Attention + add&Norm + feed forward + add&Norm
        d_model: mlp 前馈神经网络的dim
        nhead: 8头注意力机制
        dim_feedforward: 前馈神经网络的维度 2048
        dropout: 0.1
        activation: 激活函数类型
        normalize_before: 是否使用先LN  False
        """
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 这个操作是把词向量和位置编码相加操作
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        src: [494, bs, 256]  backbone输入下采样32倍后 再 压缩维度到256的特征图
        src_mask: None
        src_key_padding_mask: [bs, 494]  记录哪些位置有pad True 没意义 不需要计算attention
        pos: [494, bs, 256]  位置编码
        """
        # 数据 + 位置编码  [494, bs, 256]
        # 这也是和原版encoder不同的地方，这里每个encoder的q和k都会加上位置编码  再用q和k计算相似度  再和v加权得到更具有全局相关性（增强后）的特征表示
        # 每用一层都加上位置编码  信息不断加强  最终得到的特征全局相全局相关性最强  原版的transformer只在输入加上位置编码  作者发现这样更好
        q = k = self.with_pos_embed(src, pos)

        # multi-head attention   [494, bs, 256]
        # q 和 k = backbone输出特征图 + 位置编码
        # v = backbone输出特征图
        # 这里对query和key增加位置编码 是因为需要在图像特征中各个位置之间计算相似度/相关性 而value作为原图像的特征 和 相关性矩阵加权，
        # 从而得到各个位置结合了全局相关性（增强后）的特征表示，所以q 和 k这种计算需要+位置编码  而v代表原图像不需要加位置编码
        # nn.MultiheadAttention: 返回两个值  第一个是自注意力层的输出  第二个是自注意力权重  这里取0
        # key_padding_mask: 记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
        #                   计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        # attn_mask: 是在Transformer中用来“防作弊”的,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        #            而在encoder中通常为None 不适用  decoder中才使用
        src2 = self.self_attn(q, k, value=src)[0]

        # add + norm + feed forward + add + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding  负责预测物体  用于建模图像当中的物体信息  在每层decoder层中不断的被refine
             [100, bs, 256]  和 query_embed形状相同  且全设置为0
        memory: [h*w, bs, 256]  Encoder输出  具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # 第一个self-attention的目的：找到图像中物体的信息 -> tgt
        # 第一个多头自注意力层：输入qkv都和Encoder无关  都来自于tgt/query embedding
        # 通过第一个self-attention  可以不断建模物体与物体之间的关系  可以知道图像当中哪些位置会存在物体  物体信息->tgt
        # query embedding  +  query_pos
        q = k = self.with_pos_embed(tgt, query_pos)

        # masked multi-head self-attention  计算query embedding的自注意力
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        # add + norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 第二个self-attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中去，更好地表征了图像中的各个物体
        # 第二个多头注意力层，也叫Encoder-Decoder self attention：key和value来自Encoder层输出   Query来自Decoder层输入
        # 第二个self-attention 可以建模图像 与 物体之间的关系
        # 根据上一步得到的tgt作为query 不断的去encoder输出的特征图中去问（q和k计算相似度）  问图像当中的物体在哪里呢？
        # 问完之后再将物体的位置信息融合encoder输出的特征图中（和v做运算）  这样我得到的v的特征就有 encoder增强后特征信息 + 物体的位置信息
        # query = query embedding  +  query_pos
        # key = encoder输出特征 + 特征位置编码
        # value = encoder输出特征
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # [100, bs, 256]
        # decoder的输出是第一个self-attention输出特征 + 第二个self-attention输出特征
        # 最终的特征：知道图像中物体与物体之间的关系 + encoder增强后的图像特征 + 图像与物体之间的关系
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        batch_size=args.batch_size,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
import math

class FousionModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,\
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,\
                 activation="relu", normalize_before=False,\
                 return_intermediate_dec=False):
        super(self).__init__()
        self.embed_dim = d_model
        # self.seq = 4
        self.layer_unimodal = 1
        self.layer_multimodal = 2  # cmu_mosei
        self.audio_linear = nn.Linear(74, d_model)
        self.visual_linear = nn.Linear(35, d_model)
        # self.transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,\
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.transformer = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, audio, visual):
        audio = self.audio_linear(audio).permute(1, 0, 2)
        for i in range(self.layer_unimodal):
            audio = self.transformer(audio)

        visual = self.visual_linear(visual).permute(1, 0, 2)
        for i in range(self.layer_unimodal):
            visual = self.transformer(visual)

        fsn = torch.zeros(self.seq, audio.size(1), self.embed_dim).cuda()
        x = torch.cat([audio, fsn], dim=0)

        for i in range(self.layer_multimodal):
            if i == 0:
                x = self.transformer(x)
                x = torch.cat([x[audio.size(0):, :, :], visual], dim=0)
                x = self.transformer(x)
            else:
                x = self.transformer(x)

        return x

import  numpy
class FousionModelEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, batch_size=1):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.seq = 4
        self.embed_dim = 256
        self.layer_multimodal = 2
        self.fsn = nn.Parameter(torch.zeros(4, batch_size, 256))
        self.position_encoding = build_position_encoding()
        self.fsn_pos_embed = nn.Parameter(torch.zeros(4, batch_size, 256))

    def forward(self, src1, src2,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos1: Optional[Tensor] = None,
                pos2: Optional[Tensor] = None):
        output1 = src1
        output2 = src2
        # fsn = torch.zeros(self.seq, output1.size(1), self.embed_dim).cuda()
        output = torch.cat([output1, self.fsn], dim=0)

        # for layer in self.layers:
        #     output = layer(output, src_mask=mask,
        #                    src_key_padding_mask=src_key_padding_mask, pos=pos)
        # temp = fsn.unsqueeze(3).permute(1,2,0,3).reshape(1,256,2,2)
        # mask = numpy.full((temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3]), False)
        # mask = torch.tensor(mask, dtype=bool).cuda()
        # mask[:,:,:,:]=False
        # posFsn = self.position_encoding(temp, mask).to(fsn.tensors.dtype)

        pos = torch.cat([pos1, self.fsn_pos_embed], dim=0)

        for i in range(self.layer_multimodal):
            if i == 0:
                for layer in self.layers:
                    output = layer(output, src_mask=mask,
                                   src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is not None:
                    output = self.norm(output)
                #x = self.transformer(x)
                output = torch.cat([output[output1.size(0):, :, :], output2], dim=0)
                pos = torch.cat([pos[pos1.size(0):, :, :], pos2], dim=0)
                #x = self.transformer(x)
                for layer in self.layers:
                    output = layer(output, src_mask=mask,
                                   src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is not None:
                    output = self.norm(output)
            else:
                continue
                # for layer in self.layers:
                #     output = layer(output, src_mask=mask,
                #                    src_key_padding_mask=src_key_padding_mask, pos=pos)
                # if self.norm is not None:
                #     output = self.norm(output)

        return output[4:,:,:], pos[4:,:,:]

def build_position_encoding():
    N_steps = 256 // 2
    # if 'sine' in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # elif 'sine' in ('v3', 'learned'):
    #     position_embedding = PositionEmbeddingLearned(N_steps)
    # else:
    #     raise ValueError(f"not supported {'sine'}")

    return position_embedding

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
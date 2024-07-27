# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import numpy
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import pickle
from util.misc import NestedTensor
import PIL.Image
from scipy import signal

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # samples: NestedTensor
        # tensors: [bs, 3, H, W]  经过masked后的图片
        # mask: [bs, H, W]  记录哪些区域是mask过的无效的False  哪些是原图有效的True
        samples = samples.to(device)

        # targets: list: bs
        # 每张图片dict 7
        # 'boxes'=[num, 4]  'labels'=num  orig_size: 原图大小  size: pad后大小  area  image_id  iscrowd
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        samples1 = None
        samplesPath1 = '../detr/coco/train-EEG/'
        b,a = signal.butter(8,[0.4,0.6],'bandpass')
        for target in targets:
            sampleid = '00000' + str(target['image_id'].item())
            samplesPath11 = samplesPath1 + sampleid[-5:] + '.pkl'
            with open(samplesPath11, 'rb') as f:
                sig = pickle.load(f)
                sigPro = []
                for i in sig:
                    sigPro.append(signal.filtfilt(b,a,i))
                sig = torch.tensor(numpy.array(sigPro), dtype=torch.float32).cuda()
                sig = sig.unsqueeze(0)
                sig = sig.unsqueeze(1)
            mask = torch.tensor(numpy.zeros((1,8,3000),dtype=bool),dtype=torch.bool).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if samples1 == None:
                samples1 = NestedTensor(sig,mask).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                samples1.mask = torch.cat((samples1.mask, mask), dim=0)
                samples1.tensors = torch.cat((samples1.tensors, sig), dim=0)

        samples2 = None
        samplesPath2 = '../detr/coco/train-EYE/'
        for target in targets:
            sampleid = '00000' + str(target['image_id'].item())
            samplesPath22 = samplesPath2 + sampleid[-5:] + '.jpg'
            image = PIL.Image.open(samplesPath22)
            image = image.resize((800,800))

            image = torch.tensor(numpy.array(image),dtype=torch.float32).cuda()
            image = image.unsqueeze(0).permute(0,3,1,2)
            mask = torch.tensor(numpy.zeros((1, 800, 800), dtype=bool),dtype=torch.bool).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if samples2 == None:
                samples2 = NestedTensor(image, mask).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                samples2.mask = torch.cat((samples2.mask, mask), dim=0)
                samples2.tensors = torch.cat((samples2.tensors, image), dim=0)

        # 前向传播
        # dict: 3
        # 0 pred_logits 分类头输出[bs, 100, 92(类别数)]
        # 1 pred_boxes 回归头输出[bs, 100, 4]
        # 3 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
        outputs = model(samples, samples1, samples2)

        # 计算损失  loss_dict: 'loss_ce' + 'loss_bbox' + 'loss_giou'    用于log日志: 'class_error' + 'cardinality_error'
        loss_dict = criterion(outputs, targets)
        # 权重系数 {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        weight_dict = criterion.weight_dict
        # 总损失 = 回归损失：loss_bbox（L1）+loss_bbox  +   分类损失：loss_ce
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()   # 梯度清零
        losses.backward()       # 反向传播计算梯度 并累加梯度
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()       # 更新参数

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        samples1 = None
        samplesPath1 = '../detr/coco/val-EEG/'
        b,a = signal.butter(8,[0.4,0.6],'bandpass')
        for target in targets:
            sampleid = '00000' + str(target['image_id'].item())
            samplesPath11 = samplesPath1 + sampleid[-5:] + '.pkl'
            with open(samplesPath11, 'rb') as f:
                sig = pickle.load(f)
                sigPro = []
                for i in sig:
                    sigPro.append(signal.filtfilt(b,a,i))
                sig = torch.tensor(numpy.array(sigPro), dtype=torch.float32).cuda()
                sig = sig.unsqueeze(0)
                sig = sig.unsqueeze(1)
            mask = torch.tensor(numpy.zeros((1,8,3000),dtype=bool),dtype=torch.bool).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if samples1 == None:
                samples1 = NestedTensor(sig,mask).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                samples1.mask = torch.cat((samples1.mask, mask), dim=0)
                samples1.tensors = torch.cat((samples1.tensors, sig), dim=0)

        samples2 = None
        samplesPath2 = '../detr/coco/val-EYE/'
        for target in targets:
            sampleid = '00000' + str(target['image_id'].item())
            samplesPath22 = samplesPath2 + sampleid[-5:] + '.jpg'
            image = PIL.Image.open(samplesPath22)
            image = image.resize((800,800))

            image = torch.tensor(numpy.array(image),dtype=torch.float32).cuda()
            image = image.unsqueeze(0).permute(0,3,1,2)
            mask = torch.tensor(numpy.zeros((1, 800, 800), dtype=bool),dtype=torch.bool).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if samples2 == None:
                samples2 = NestedTensor(image, mask).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                samples2.mask = torch.cat((samples2.mask, mask), dim=0)
                samples2.tensors = torch.cat((samples2.tensors, image), dim=0)

        # 前向传播
        # dict: 3
        # 0 pred_logits 分类头输出[bs, 100, 92(类别数)]
        # 1 pred_boxes 回归头输出[bs, 100, 4]
        # 3 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
        outputs = model(samples,samples1,samples2)

        # 计算损失 显示log
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # 后处理
        # orig_target_sizes = [bs, 2]  bs张图片的原图大小
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # list: bs    每个list都是一个dict  包括'scores'  'labels'  'boxes'三个字段
        # scores = Tensor[100,]  这张图片预测的100个预测框概率分数
        # labels = Tensor[100,]  这张图片预测的100个预测框所属类别idx
        # boxes = Tensor[100, 4] 这张图片预测的100个预测框的绝对位置坐标(相对这张图片的原图大小的坐标)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # 分割的内容不执行
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        # dict: 2
        # img_idx = dict{3} = 'score'[100,] + 'labels'[100,] + 'boxes'[100,4]
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

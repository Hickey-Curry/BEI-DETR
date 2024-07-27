# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # 其他参数的学习率
    parser.add_argument('--lr', default=1e-4, type=float)
    # backbone参数的学习率
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters  是否冻结bn层
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    # backbone是否使用空洞卷积
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # 使用正余弦位置编码还是可学习的位置编码
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    # backbone输入transformer特征的维度
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    # transformer中隐藏层的维度
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    # 多头注意力用几头
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=15, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation   实例分割的参数
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss   是否使用辅助loss  是否使用其他层decoder一起计算loss 默认是不使用
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher 匹配器的损失比重
    # 分类损失权重  1
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    # L1回归损失权重  5
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    # Giou回归损失权重  2
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients  真正的loss比重
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    # 数据集的类型
    parser.add_argument('--dataset_file', default='coco')
    # 数据集root
    parser.add_argument('--coco_path', type=str, default='../detr/coco')
    # 是否在训练目标检测任务的同时进行全局分割
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='outputs_bs2_1bb_2e4d_15q',
    #parser.add_argument('--output_dir', default='outputs_bs2_2bb_4e6d',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    #parser.add_argument('--resume', default='./outputs_bs2_2bb_4e6d/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # ------------------------------------------ 一些基本的参数 ------------------------------------------
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir="outputs_bs2_1bb_2e4d_15q/experiment")

    # 初始化分布式训练
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # 是否需要冻结backbone训练
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility   固定随机数种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ------------------------------------------ 一些基本的参数 ----------------------------------------

    # ------------------------------------------ 模型部分 ------------------------------------------
    # 搭建模型
    # model: 整体模型
    # criterion: 损失函数
    # postprocessors: bbox后处理  调用coco api
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # DDP分布式训练
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 打印模型参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # ------------------------------------------ 模型部分 ------------------------------------------

    # ------------------------------------------ 学习率策略和优化器部分 ------------------------------------------
    # 对backbone和transformer、ffn分别设置不同学习率
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    # 优化器和学习率调整策略
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # ------------------------------------------ 学习率策略和优化器部分 ------------------------------------------

    # ------------------------------------------ 数据集部分 ------------------------------------------
    # 创建训练和验证数据集
    dataset_train = build_dataset(image_set='train', args=args, dim=0)
    dataset_val = build_dataset(image_set='val', args=args, dim=0)

    # 定义数据集采样策略
    if args.distributed:
        """
        如果使用分布式训练，则使用 DistributedSampler 对训练集和验证集进行采样
        DistributedSampler 可以在多个进程之间进行数据分发和负载均衡
        """
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        """
        如果不使用分布式训练，则使用 torch.utils.data.RandomSampler 对训练集进行采样
        RandomSampler 可以随机打乱数据集的顺序
        """
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        """
        使用 torch.utils.data.SequentialSampler 对验证集进行采样
        SequentialSampler 可以按照数据集的原始顺序进行采样
        """
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    """
    创建一个 BatchSampler 对象，用于将采样得到的样本组合成批次
    使用的采样器是之前定义的 sampler_train
    批次大小由 args.batch_size 指定
    drop_last=True 表示如果最后一个批次的大小不足 args.batch_size，那么就忽略这个批次
    """
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    """
    创建一个 DataLoader 对象，用于加载训练数据集
    使用的批次采样器是之前定义的 batch_sampler_train
    数据集是之前定义的 dataset_train
    collate_fn 指定了如何将一个批次的样本组合成一个张量
    num_workers 指定了用于数据加载的线程数
    """
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    """
    创建一个 DataLoader 对象，用于加载验证数据集
    使用的批次采样器是之前定义的 sampler_val
    数据集是之前定义的 dataset_val
    collate_fn 指定了如何将一个批次的样本组合成一个张量
    num_workers 指定了用于数据加载的线程数
    """
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=True,
                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # 是否加载全景分割数据
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    # ------------------------------------------- 数据集部分 --------------------------------------------

    # -------------------------------------------------- 训练前的准备 ----------------------------------------------
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    # 是否断点续训
    if args.resume:
        # 如果resume参数的值是以'https'开头的，说明是一个URL
        if args.resume.startswith('https'):
            # 从URL加载模型的状态字典（包括模型的参数等）
            # map_location='cpu'表示将模型加载到CPU上，check_hash=True表示验证下载内容的哈希值以确保内容完整性
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 如果resume参数的值不是URL，则直接当作本地文件路径来加载模型状态字典
            checkpoint = torch.load(args.resume, map_location='cpu')

        # 将加载的模型状态字典中的'model'部分加载到model_without_ddp中
        # model_without_ddp可能是没有使用数据并行（如DistributedDataParallel）的模型
        model_without_ddp.load_state_dict(checkpoint['model'])
        # 如果不是在评估模式（args.eval为False），并且checkpoint中包含优化器、学习率调度器和轮数的信息
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # 将优化器的状态加载回optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 将学习率调度器的状态加载回lr_scheduler
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # 更新开始训练的轮数，从checkpoint中保存的轮数加一开始
            args.start_epoch = checkpoint['epoch'] + 1

    # 是否需要评估
    if args.eval:
        # 根据args中的参数构建测试数据集
        dataset_test = build_dataset(image_set='test', args=args)

        # 根据是否使用分布式训练，构建不同的采样器
        # 分布式训练时，使用DistributedSampler进行采样
        if args.distributed:
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            # 非分布式训练时，使用SequentialSampler按顺序采样
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        # 构建数据加载器，用于加载测试数据
        data_loader_test = DataLoader(dataset_val, args.batch_size, sampler=sampler_test,
                                      drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # 执行模型评估，并返回测试统计信息和COCO评估器
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_test, base_ds, device, args.output_dir)
        # 如果设置了args.output_dir，表示需要保存评估结果
        if args.output_dir:
            # 调用utils.save_on_master函数，将COCO评估的bbox结果保存在指定的输出目录下
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    # ------------------------------------------------ 训练前的准备 ----------------------------------------------------

    # ---------------------------------------------- 开始训练 ----------------------------------------------

    best_map=0.0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 分布式训练
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # 训练一个epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()  # 调整学习率

        # 保存模型
        if args.output_dir:
            # 初始化一个列表，用于存储检查点文件的路径
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            # 在学习率下降前，以及每100个训练轮次时，额外保存检查点
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                # 根据当前训练轮次，生成形如'checkpoint0001.pth'的文件名，并添加到检查点文件路径列表中
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            # 遍历每一个检查点文件路径
            for checkpoint_path in checkpoint_paths:
                # 在主进程上保存模型状态、优化器状态、学习率调度器状态、当前轮次以及训练参数
                # 这些信息被保存为一个字典，并作为utils.save_on_master函数的第二个参数
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),     # 保存模型的参数
                    'optimizer': optimizer.state_dict(),         # 保存优化器的状态
                    'lr_scheduler': lr_scheduler.state_dict(),   # 保存学习率调度器的状态
                    'epoch': epoch,                              # 保存当前的训练轮次
                    'args': args,                                # 保存训练参数或配置
                }, checkpoint_path)                              # 将这些信息保存到指定的检查点文件路径中

        # 调用evaluate函数进行模型评估
        # 评估函数将返回测试统计信息和COCO评估器对象
        test_stats, coco_evaluator = evaluate(
            model,           # 传入待评估的模型
            criterion,       # 传入损失函数，用于计算模型预测与真实标签之间的差异
            postprocessors,  # 传入后处理函数或函数列表，用于对模型输出进行后处理
            data_loader_val, # 传入验证集的数据加载器，用于加载验证数据
            base_ds,         # 传入基础数据集对象，可能用于某些评估逻辑
            device,          # 传入设备信息，指定模型在哪个设备上运行（如'cuda'或'cpu'）
            args.output_dir  # 传入输出目录路径，用于保存评估结果或其他文件
        )

        # 保存日志
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        tags = ["lr", "train_class_error", "train_loss", "train_loss_bbox", "test_class_error", "test_loss", "test_loss_bbox", "mAP"]
        tb_writer.add_scalar(tags[0], train_stats["lr"], epoch)
        tb_writer.add_scalar(tags[1], train_stats["class_error"], epoch)
        tb_writer.add_scalar(tags[2], train_stats["loss"], epoch)
        tb_writer.add_scalar(tags[3], train_stats["loss_bbox"], epoch)
        tb_writer.add_scalar(tags[4], test_stats["class_error"], epoch)
        tb_writer.add_scalar(tags[5], test_stats["loss"], epoch)
        tb_writer.add_scalar(tags[6], test_stats["loss_bbox"], epoch)
        tb_writer.add_scalar(tags[7], test_stats["coco_eval_bbox"][1], epoch)

        # 如果设置了输出目录，并且当前进程是主进程（在分布式训练中通常只有一个主进程负责保存文件）
        if args.output_dir and utils.is_main_process():
            # 打开名为"log.txt"的文件，以追加模式写入日志统计信息
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            # 如果是评估日志
            if coco_evaluator is not None:
                # 创建一个名为'eval'的目录，如果目录已存在则不报错
                (output_dir / 'eval').mkdir(exist_ok=True)

                # 如果COCO评估器包含bbox评估结果
                if "bbox" in coco_evaluator.coco_eval:
                    # 初始化一个文件名列表，包含'latest.pth'
                    filenames = ['latest.pth']
                    # 如果当前训练轮次是50的倍数
                    if epoch % 50 == 0:
                        # 添加形如'000.pth'的文件名到列表中
                        filenames.append(f'{epoch:03}.pth')
                    # 遍历文件名列表
                    for name in filenames:
                        # 将bbox评估结果保存到输出目录下的'eval'子目录中
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

            # 如果当前的评估结果中的mAP（mean Average Precision，平均精度均值）高于之前记录的最佳mAP
            if test_stats["coco_eval_bbox"][1] >best_map:
                # 更新最佳mAP值
                best_map = test_stats["coco_eval_bbox"][1]
                # 在主进程上保存模型状态、优化器状态、学习率调度器状态、当前轮次以及参数
                utils.save_on_master({
                        'model': model_without_ddp.state_dict(),  # 保存模型参数
                        'optimizer': optimizer.state_dict(),      # 保存优化器状态
                        'lr_scheduler': lr_scheduler.state_dict(),# 保存学习率调度器状态
                        'epoch': epoch,                           # 保存当前训练轮次
                        'args': args,                             # 保存命令行参数或配置
                    }, output_dir / 'best.pth')

    #输出训练的时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # ---------------------------------------------- 训练结束 ----------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from __future__ import division
import pickle

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, wrap_fp16_model
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from mmcv.utils import TORCH_VERSION, digit_version

import sys

sys.path.append("..")
from collections import OrderedDict


def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    delet_modules = ['UpConv_4x', 'UpConv_2x', 'UpConv_1x', 'DwConv_2x', 'neck']
    for name, param in model.named_parameters():

        #print(name)
        if 'img_backbone' not in name:  # only calculate the backbone
            continue
        contains = any(substring in name for substring in delet_modules)
        if contains:
            continue
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.info("tunable:" + name)

    all_param = (all_param - trainable_params) / 1e6
    trainable_params = trainable_params / 1e6

    if all_param == 0:
        logger.info(
            f"trainable params: {trainable_params} || all params: {trainable_params} || trainable%: {100 :.2f}"
        )
    else:
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.core.apis.train import custom_train_model
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2'  # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    model.init_weights()

    # if load v99
    try:
        if 'load_from_v99' in cfg:
            print("\n****************************************************************\n")
            print("load from v99")
            print("\n****************************************************************\n")
            if not os.path.exists(cfg.get('load_from_v99')):
                logger.info('no v99 checkpoint')
                exit()
            with open(cfg.get('load_from_v99'), "rb") as f:
                state_dict = torch.load(f)['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('img_neck.', 'neck.')  # remove `img_neck.` (wrong size)
                    new_state_dict[name] = v

                incompatible = model.load_state_dict(new_state_dict, strict=False)
                logger.info(incompatible)
            print("\n****************************************************************\n")
            print("end load v99")
            print("\n****************************************************************\n")
    except:
        print("\n****************************************************************\n")
        print("no need to load v99")
        print("\n****************************************************************\n")

    # load pretrained
    if 'sam_vit' in args.config or 'vitdet' in args.config:
        print("\n****************************************************************\n")
        print("load vit checkpoint")
        print("\n****************************************************************\n")

        if not os.path.exists(cfg.get('load_from')):
            logger.info('no checkpoint')
            exit()

        if cfg.get('load_from').endswith(".pkl"):
            with open(cfg.get('load_from'), "rb") as f:
                state_dict = pickle.load(f, encoding="latin1")
                f.close()
            if "model" in state_dict and "__author__" in state_dict:
                # file is in Detectron2 model zoo format
                logger.info("Reading a file from '{}'".format(state_dict["__author__"]))
                state_dict = state_dict['model']
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                # if "blobs" in state_dict:
                #     # Detection models have "blobs", but ImageNet models don't
                #     data = state_dict["blobs"]
                # data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                # return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
                exit()
        else:
            # SAM pretrained
            with open(cfg.get('load_from'), "rb") as f:
                state_dict = torch.load(f)
                # state_dict = state_dict['state_dict']

                # MAE pretrained
                if 'mae' in cfg.get('load_from'):
                    state_dict = state_dict['model']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'mae' in cfg.get('load_from'):
                name = 'img_backbone.' + k  # remove `module.`
                new_state_dict[name] = v
                logger.info(name)

            elif 'image_encoder' in k or 'img_backbone.' in k:
                name = k.replace('image_encoder.', 'img_backbone.')  # remove `module.`
                new_state_dict[name] = v

            elif 'final_' in cfg.get('load_from'):
                name = k.replace('backbone.net.', 'img_backbone.')  # remove `module.`
                name = name.replace('backbone.simfp', 'img_backbone.simfp')
                new_state_dict[name] = torch.from_numpy(v)

        incompatible = model.load_state_dict(new_state_dict, strict=False)
        logger.info(incompatible)
        # logger.info("\nmissing_keys:\n  {}".incompatible.missing_keys)
        # logger.info("\nunexpected_keys:\n {}".incompatible.unexpected_keys)
        logger.info('load end')

        if 'fixed_backbone' in cfg.get('model').img_backbone:
            if cfg.get('model').img_backbone.fixed_backbone:
                logger.info('fixed the backbone')
                visual_weights = {k: v.float() for k, v in new_state_dict.items()}

                for name, param in model.named_parameters():
                    if name in visual_weights:
                        if 'pos_embed' in name and not cfg.get('model').img_backbone.fixed_pos_embed:
                            continue  # not fixed position embedding
                        if 'neck' in name and not cfg.get('model').img_backbone.fixed_sam_neck:
                            continue  # not fixed sam neck
                        # logger.info(name)
                        if 'fixed_bias' in cfg.get('model').img_backbone and ('bias' in name or 'norm' in name):
                            if cfg.get('model').img_backbone.fixed_bias:
                                param.requires_grad = False
                                logger.info(name)
                        else:
                            param.requires_grad = False
                            logger.info(name)

        print_trainable_parameters(model, logger)

    if cfg.get('SyncBN', False):
        import torch.nn as nn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Using SyncBN")

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork', force=True)
    main()

# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.datasets import build_fmow_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import models_mae
import models_mae_group_channels

import models_mae_spectral

from engine_pretrain import train_one_epoch, train_one_epoch_temporal

import warnings

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='group_c',
                        choices=['group_c', 'tensor', ],
                        help='Use channel model')
    parser.add_argument('--model', default='mae_vit_large_patch16_128', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=96, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--spatial_mask', action='store_true', default=False,
                        help='Whether to mask all channels of a spatial location. Only for indp c model')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_path', default='', type=str,
                        help='Train.csv path')
    parser.add_argument('--dataset_type', default='sentinel',
                        choices=['rgb',  'sentinel', 'euro_sat', 'bigearthnet'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--masked_bands', type=int, nargs='+', default=None,
                        help='Sequence of band indices to mask (with mean val) in sentinel dataset')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Which bands (0 indexed) to drop from sentinel data.")
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC mae")

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_different_size', default='',
                        help='continue to pretrain for different size')
    parser.add_argument('--wandb', type=str, default=None,
                        help="Wandb project name, eg: sentinel_pretrain")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=os.getenv('LOCAL_RANK', 0), type=int)  # prev default was -1
    # parser.add_argument('--local-rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_fmow_dataset(is_train=True, args=args)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=False, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # prefetch_factor=8,

    )

    # define the model
    if args.model_type == 'group_c':
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
        print(f"Grouping bands {args.grouped_bands}")
        model = models_mae_group_channels.__dict__[args.model](img_size=args.input_size,
                                                               patch_size=args.patch_size,
                                                               in_chans=dataset_train.in_c,
                                                               channel_groups=args.grouped_bands,
                                                               spatial_mask=args.spatial_mask,
                                                               norm_pix_loss=args.norm_pix_loss)
    elif args.model_type == 'tensor':
        model = models_mae_spectral.__dict__[args.model]()

    else:
        model = models_mae.__dict__[args.model](img_size=args.input_size,
                                                patch_size=args.patch_size,
                                                in_chans=dataset_train.in_c,
                                                norm_pix_loss=args.norm_pix_loss)
    if args.resume_different_size:
        checkpoint = torch.load(args.resume_different_size, map_location='cpu')

        print("Load pre-trained_size checkpoint from: %s" % args.resume_different_size)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['pos_embed_spatial', 'pos_embed', 'decoder_pos_embed', 'patch_embed.proj.weight',
                  'decoder_pos_embed_spatial', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            # for k in ['pos_embed', 'decoder_pos_embed', 'patch_embed.proj.weight',
            #                'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint['model'] and checkpoint['model'][k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]

        # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint['model'])
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # misc.load_model_different_size(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, entity="mae-sentinel")
        wandb.config.update(args)
        wandb.watch(model)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.model_type == 'temporal':
            train_stats = train_one_epoch_temporal(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        else:
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )

        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "new_fmow_sep.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            try:
                wandb.log(log_stats)
            except ValueError:
                print(f"Invalid stats?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

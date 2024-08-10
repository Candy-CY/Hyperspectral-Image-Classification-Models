# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import wandb
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from sklearn.metrics import average_precision_score, precision_score
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # print(targets.shape)
        samples = samples.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)


        #

        if mixup_fn is not None and args.dataset_type != 'bigearthnet_finetune':
            samples, targets = mixup_fn(samples, targets)
            # print(targets)
            # print(targets.shape)
            # print(targets.dtype)
            # if args.smoothing > 0.:
            #     targets = torch.topk(targets, k=1, dim=1).indices.squeeze(1)


        with torch.cuda.amp.autocast():
            outputs = model(samples)
            # outputs = model(optical_images=samples)
            # print(outputs)
            # print(outputs.shape)

            loss = criterion(outputs, targets)
            # print('targets')

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device,args):
    # criterion = torch.nn.CrossEntropyLoss()
    if args.smoothing > 0.:
        # criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # criterion = SoftTargetCrossEntropy()
        criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # output = model(optical_images=images)
            loss = criterion(output, target)
    #     if args.dataset_type == 'bigearthnet_finetune':
    #         output_cpu = output.cpu()
    #         target_cpu = target.cpu()
    #
    #         output_true = torch.gt(output_cpu, 0.5)
    #         output_score = torch.where(output_true, torch.tensor(1), torch.tensor(0))
    #         map = precision_score(output_score, target_cpu, average='macro')
    #         batch_size = images.shape[0]
    #         metric_logger.update(loss=loss.item())
    #         metric_logger.meters['mAP'].update(map.item(), n=batch_size)
    # metric_logger.synchronize_between_processes()
    # print(' Map {map.global_avg:.3f} loss {losses.global_avg:.3f}'
    #               .format(map=metric_logger.mAP, losses=metric_logger.loss))

        # if args.dataset_type != 'bigearthnet_finetune':
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


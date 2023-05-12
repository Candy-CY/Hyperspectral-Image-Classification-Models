import os
import argparse
import time
import apex
import logging
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.multiprocessing
import torch.distributed as dist
from models.sync_batchnorm.replicate import patch_replication_callback
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
import scipy.io as scio


def get_logger(save_path):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    log_format = '%(asctime)s %(message)s'
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main_process(args):
    return not args.distributed == 'True' or (args.distributed == 'True' and args.rank % args.world_size == 0)

class all_loss(nn.Module):
    def __init__(self):
        super().__init__()

        # ce
        self.criterion_1 = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion_2 = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, output, target):
        
        if len(output) == 2:
            output_1, output_2 = output

            target = target.cuda(non_blocking=True).long()

            loss1 = self.criterion_1(output_1, target)
            loss2 = self.criterion_2(output_2, target)

#            loss = loss1 + 0.4*loss2
            
            return loss1, loss2
            
        else:
       
            output_1 = output

            target = target.cuda(non_blocking=True).long()

            loss1 = self.criterion_1(output_1, target)

            loss = loss1

            return loss

class Trainer(object):
    def __init__(self, args, LOCAL_RANK=0):
        self.args = args

        # Define Saver
        if main_process(args):
            self.saver = Saver(args)
            self.saver.save_experiment_config()
            self.logger = get_logger(self.saver.experiment_dir)

        # print args
        if main_process(args):
            self.logger.info(args)

        # Define Dataloader
        if 'WHUHi' in self.args.dataset:
            from dataloaders.datasets.WHU_Hi import make_data_loader
            self.train_loader, self.val_loader = make_data_loader(args)
        else:
            raise NotImplementedError

        if 'WHUHi' in self.args.dataset:
            in_channels = 3
        else:
            in_channels = 3

        if 'LongKou' in self.args.dataset:
            classes = 9
        elif 'HanChuan' in self.args.dataset:
            classes = 16
        elif 'HongHu' in self.args.dataset:
            classes = 22
        else:
            raise NotImplementedError

        # Define model

        from models.network_local_global import rat_model

        model = rat_model(args, classes, in_channels)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr*10}]

        model.cuda()

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay, nesterov=args.nesterov)

        #optimizer = torch.optim.Adam(train_params,weight_decay=args.weight_decay)

        # Define Criterion

        self.criterion = all_loss()

        # Define Evaluator

        self.evaluator = Evaluator(classes)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args, args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Define accuracy

        if 'WHUHi' in self.args.dataset:

            self.val_vote_acc = []
            self.evaluator_vote = Evaluator(classes)
        else:
            self.val_vote_acc = None

        # Define train form

        if args.distributed == 'True':
            if args.use_apex == 'True':  # nvidia 的 apex
                model = apex.parallel.convert_syncbn_model(model)
                model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.opt_level)
                model = apex.parallel.DistributedDataParallel(model)
                if main_process(args):
                    self.logger.info("Implementing distributed hybrid training!")
            else:  # pytorch official
                model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[LOCAL_RANK],find_unused_parameters=True)
                if main_process(args):
                    self.logger.info("Implementing distributed training!")
        else:
            if args.use_apex == 'True':  # nvidia 的 apex
                model = apex.parallel.convert_syncbn_model(model)
                model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.opt_level)
                if main_process(args):
                    self.logger.info("Implementing parallel hybrid training!")
            model = torch.nn.DataParallel(model.cuda())  # 普通的单机多卡
            patch_replication_callback(model)
            if main_process(args):
                self.logger.info("Implementing parallel training!")

        self.model, self.optimizer = model, optimizer

        # Resuming checkpoint
        self.best_pred = 0.0

        if args.ft=='True':
            if not os.path.isfile(args.resume):
                if main_process(args):
                    raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            else:
                if main_process(args):
                    self.logger.info("=> loading ft model...")

                checkpoint = torch.load(args.resume, map_location='cpu')
                ckpt_dict = checkpoint['state_dict']
                model_dict = {}

                state_dict = model.state_dict()
                for k, v in ckpt_dict.items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)

                if args.distributed=='True':
                    self.model.load_state_dict(state_dict)
                else:
                    self.model.module.load_state_dict(state_dict)

                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # self.best_pred = checkpoint['best_pred']
                if main_process(args):
                    self.logger.info("=> loaded checkpoint '{}' (epoch {})"
                                     .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft == 'True':
            self.args.start_epoch = 0

    def training(self, epoch, args, trn_loss):
        epoch_loss = 0.0
        start_time = time.time()
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image = sample['image']
            target = sample['label']
            image = image.cuda(non_blocking=True)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            output = self.model(image)
            loss1, loss2 = self.criterion(output, target)
            
            loss = loss1 + 0.4*loss2

            reduced_loss = loss.data.clone()
            
            trn_loss.append(loss1.item())

            if self.args.distributed == 'True':
                reduced_loss = reduced_loss / args.world_size
                dist.all_reduce_multigpu([reduced_loss])

            self.optimizer.zero_grad()

            epoch_loss += loss.item()

            if self.args.use_apex == 'True':
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=20, norm_type=2)

            self.optimizer.step()

            if main_process(self.args):
                tbar.set_description('Training batch: %d' % (i + 1))
                tbar.set_postfix(Loss=epoch_loss / (i + 1))

        end_time = time.time()

        if main_process(self.args):
            self.logger.info('Training epoch [{}/{}]: Loss: {:.4f}. Cost {:.4f} secs'.format(epoch+1, self.args.epochs, epoch_loss*1.0/(i+1),end_time-start_time))
            
        return trn_loss

    def validation(self, epoch, val_loss, isbest=True):
        if main_process(self.args):
            self.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.model.eval()
        self.evaluator.reset()

        if 'WHUHi' in self.args.dataset:
            vote_prob = 0
            
        if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
            preds = []

        tbar = tqdm(self.val_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image = sample['image']
            target = sample['label']
            image = image.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(image)
                loss = self.criterion(output, target)
                val_loss.append(loss.item())
                
            if main_process(self.args):
                tbar.set_description('Validation batch: %d' % (epoch))
                
            if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
                preds.append(output.cpu().numpy().argmax(axis=1)) # b, h, w
                
            if 'WHUHi' in self.args.dataset and self.args.mode == 'soft':
                vote_prob += output.cpu().numpy() # 1,c,h,w
                            
        if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
            preds = np.concatenate(preds, axis=0).astype('int') # B, h, w 
            _, h, w = preds.shape
            vote_pred = np.zeros([1, h, w]).astype('int')
            
            for ii in range(h):
                for jj in range(w):
                    vote_pred[0, ii, jj] = np.argmax(np.bincount(preds[:,ii,jj]))
                
        if 'WHUHi' in self.args.dataset and self.args.mode == 'soft':
            vote_pred = np.argmax(vote_prob, axis=1) # 1,h,w
        
        target = target.cpu().numpy()  # batch_size * 256 * 256
        self.evaluator.add_batch(target, vote_pred)

        if 'WHUHi' in self.args.dataset:
            OA = self.evaluator.Pixel_Accuracy()
            mIOU, IOU = self.evaluator.Mean_Intersection_over_Union()
            mAcc, Acc = self.evaluator.Pixel_Accuracy_Class()
            Kappa = self.evaluator.Kappa()
            self.val_vote_acc.append(OA)
            if main_process(self.args):
                self.logger.info('[Val Vote: OA: %.4f]' % (OA))

        if main_process(self.args):

            self.logger.info('[Epoch: %d, Val OA: %.4f, mIOU: %.4f, mAcc: %.4f, Kappa: %.4f]' % (
            epoch, OA, mIOU, mAcc, Kappa))

            if 'WHUHi' in self.args.dataset:
                new_pred = OA
            else:
                raise NotImplementedError

            if (new_pred > self.best_pred and isbest==True) or isbest==False:

                self.best_pred = new_pred

                if self.args.distributed=='True':
                    self.saver.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                    }, val_vote_acc=self.val_vote_acc,is_best=isbest)
                else:
                    self.saver.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                    }, val_vote_acc=self.val_vote_acc ,is_best=isbest)
                    
        return val_loss


def main():
    parser = argparse.ArgumentParser(description="Gaofen Challenge Training")
    '''
        Model
    '''
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18','resnet50', 'vgg16', 'hrnet18', 'vitaev2_s','mobilenetv2','swint'],
                        help='backbone name')
    '''
        Dataset
    '''
    ## WHUHi + district + channel + sample
    ## eg: WHUHi_LongKou_10_100

    parser.add_argument('--dataset', type=str, default=None,help='dataset name')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='crop image size')

    '''
        Hyper Parameters
    '''
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--groups', type=int, default=120,
                        help='number of regions')
    parser.add_argument('--ra_head_num', type=int, default=None,
                        help='number of regions')
    parser.add_argument('--ga_head_num', type=int, default=None,
                        help='number of regions')
    '''
        Optimizer
    '''
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    '''
        Fine-tune
    '''
    parser.add_argument('--ft', type=str, default='False',
                        choices=['True', 'False'],
                        help='finetuning on a different dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--freeze_bn', action='store_true', default=False,
                        help='whether freeze bn while finetuning')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='whether freeze backbone while finetuning')
    '''
        Evaluation
    '''
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    '''
        Others
    '''
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    '''
        apex
    '''
    parser.add_argument('--use_apex', type=str, default='False',
                        choices=['True', 'False'], help='use apex')
    parser.add_argument('--opt_level', type=str, default='O0',
                        choices=['O0', 'O1', 'O2', 'O3'], help='hybrid training')
    '''
        distributed
    '''
    parser.add_argument('--distributed', type=str, default='True',
                        choices=['True', 'False'], help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    
    '''
        mode
    '''
    parser.add_argument('--mode', type=str, default='soft',
                        choices=['soft', 'hard'], help='voting mode')

    args = parser.parse_args()

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.distributed == 'True':
        #args.world_size = int(os.environ['SLURM_NTASKS'])
        args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        #args.rank = int(os.environ['SLURM_PROCID'])
        args.rank = int(os.environ["RANK"])
        #LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
        LOCAL_RANK = int(os.environ['LOCAL_RANK']) #args.rank % torch.cuda.device_count()
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)#分布式TCP初始化
        torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数

        trainer = Trainer(args,LOCAL_RANK)
    else:
        trainer = Trainer(args)
    
    trn_loss = []
    val_loss = []
    trn_time = 0
    
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trn_time1 = time.time()
        trn_loss = trainer.training(epoch, args, trn_loss)
        trn_time2 = time.time()
        if not trainer.args.no_val and epoch % args.eval_interval == 0:
            val_loss = trainer.validation(epoch, val_loss)
            
        trn_time = trn_time + trn_time2 - trn_time1
            
    
    scio.savemat(os.path.join(trainer.saver.experiment_dir,'trn_loss_iters_{}_{}_loss1.mat'.format(args.dataset,args.mode)),{'data': trn_loss})
    
    scio.savemat(os.path.join(trainer.saver.experiment_dir,'val_loss_epochs_{}_{}_loss1.mat'.format(args.dataset,args.mode)),{'data': val_loss})
            
    val_acc = trainer.val_vote_acc
    
    scio.savemat(os.path.join(trainer.saver.experiment_dir,'val_acc_epochs_{}_{}.mat'.format(args.dataset,args.mode)),{'data': val_acc})
    
    val_acc = np.array(val_acc)
    
    tes_time1 = time.time()

    trainer.validation(epoch, val_loss, isbest=False)
    
    tes_time2 = time.time()
    
    tes_time = tes_time2 - tes_time1
    
    trainer.logger.info('[Trn time: %.4f]' % (trn_time))
    trainer.logger.info('[Tes time: %.4f]' % (tes_time))
    
    

if __name__ == '__main__':
    main()

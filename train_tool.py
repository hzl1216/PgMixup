import time
import os

import torch.nn as nn
import torch
import logging
from utils import *
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

LOG = logging.getLogger('main')
args = None


def set_args(input_args):
    global args
    args = input_args


def train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model,optimizer, ema_optimizer, all_labels,epoch, scheduler=None):
    labeled_train_iter = iter(train_labeled_loader)
    unlabeled_train_iter = iter(train_unlabeled_loader)

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()
    end = time.time()
    for i in range(args.epoch_iteration):
        try:
            inputs_x, targets_x, label_index = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_labeled_loader)
            inputs_x, targets_x, label_index = labeled_train_iter.next()

        try:
            (inputs_aug, inputs_std), _, unlabel_index = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(train_unlabeled_loader)
            (inputs_aug, inputs_std), _, unlabel_index = unlabeled_train_iter.next()

        # measure data loading time
        meters.update('data_time', time.time() - end)
        inputs_x = inputs_x.cuda()
        inputs_aug = inputs_aug.cuda()
        inputs_std = inputs_std.cuda()

        batch_size = inputs_x.size(0)
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1), 1).cuda(non_blocking=True)

        targets_u = ema_model(inputs_std)
        if args.softmax_temp > 0:
            targets_u = targets_u / args.softmax_temp
        targets_u = torch.softmax(targets_u, dim=1)
        targets_u = targets_u.detach()
        all_inputs = torch.cat([inputs_aug, inputs_std], dim=0)
        if args.mixup:
            length = get_unsup_size(epoch + i / args.epoch_iteration)
            input_a = torch.cat([inputs_x, inputs_std[: length]], dim=0)
            target_a = torch.cat([targets_x, targets_u[: length]], dim=0)
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            idx = torch.randperm(input_a.size(0))
            input_b = input_a[idx]
            target_b = target_a[idx]
            mixed_inputs = l * input_a + (1 - l) * input_b
            mixed_targets = l * target_a + (1 - l) * target_b
            del input_a, target_a, input_b, target_b
            all_inputs = torch.cat([all_inputs, mixed_inputs], dim=0)
            all_logits = model(all_inputs)
            logits_aug, logits_std = all_logits[:args.batch_size * args.unsup_ratio * 2].chunk(2)
            if args.softmax_temp > 0:
                logits_std = logits_std / args.softmax_temp
            logits_mixup = all_logits[args.batch_size * args.unsup_ratio * 2:]
            del all_logits
            loss, class_loss, consistency_loss = semiloss_mixup(logits_mixup, mixed_targets, logits_aug,
                                                                logits_std.detach())
        else:
            logits_x = model(inputs_x)
            logits_u = model(inputs_aug)
            loss, class_loss, consistency_loss = semiloss(logits_x, targets_x, logits_aug, logits_std.detach())
        meters.update('loss', loss.item())
        meters.update('class_loss', class_loss.item())
        meters.update('cons_loss', consistency_loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'.format(
                    epoch, i, args.epoch_iteration, meters=meters))


    ema_optimizer.step(bn=True)
    return meters.averages()['class_loss/avg'], meters.averages()['cons_loss/avg'], all_labels


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    all_labels = None
    all_logits = None
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            logits = model(inputs)
            loss = criterion(logits, targets)

            if all_labels is None:
                all_labels = targets
                all_logits = logits
            else:
                all_labels = torch.cat([all_labels, targets], dim=0)
                all_logits = torch.cat([all_logits, logits], dim=0)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if batch_idx % args.print_freq == 0:
                print(
                    '{batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
#    conf_matrix = confusion_matrix(all_logits, all_labels)
#    plot_confusion_matrix(conf_matrix.numpy(), epoch)
    return losses.avg, top1.avg


class WeightEMA(object):
    def __init__(self, model, ema_model, tmp_model=None, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        if tmp_model is not None:
            self.tmp_model = tmp_model.cuda()
        self.wd = args.weight_decay

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                if args.optimizer == 'Adam':
                    param.data.mul_(1 - self.wd)

def save_checkpoint(name ,state, dirpath, epoch):
    filename = '%s_%d.ckpt' % (name, epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_entropy_weight(epoch):
    if epoch > args.consistency_rampup:
        return args.entropy_cost
    else:
        return 0


def softmax_temp(logits):
    if args.softmax_temp > 0:
        logits = logits / args.softmax_temp
    logits = torch.softmax(logits,dim=1)
    return logits


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total,alpha=0.004, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.alpha = alpha
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        cosine_decay = 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return decayed
    


def mixup(all_inputs, all_targets, model,epoch):
    l = np.random.beta(args.alpha, args.alpha)

    length = get_unsup_size(epoch)
    all_inputs = all_inputs[:args.batch_size+length]
    all_targets = all_targets[:args.batch_size+length]
    idx = torch.randperm(all_inputs.size(0))
    input_a, input_b = all_inputs[idx], all_inputs[idx][idx]
    target_a, target_b = all_targets[idx], all_targets[idx][idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    logits = model(mixed_input)
    
    return logits, mixed_target


def semiloss(logits_x, targets_x, logits_u, targets_u):
    class_loss = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))
    consistency_loss = torch.mean(torch.sum(F.softmax(targets_u,1) * (F.log_softmax(targets_u, 1) - F.log_softmax(logits_u, dim=1)), 1))

    return class_loss + args.consistency_weight*consistency_loss, class_loss, consistency_loss


def semiloss_mixup(logits_x, targets_x, logits_u, targets_u):
    class_loss = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))
    if args.confidence_thresh > 0:
        loss_mask = torch.max(torch.softmax(targets_u, dim=1),dim=1)[0].gt(args.confidence_thresh).float().detach()
        consistency_loss = torch.mean(torch.sum(F.softmax(targets_u,1) * (F.log_softmax(targets_u, 1) - F.log_softmax(logits_u, dim=1)), 1)*loss_mask)
    else:
        consistency_loss = torch.mean(torch.sum(F.softmax(targets_u,1) * (F.log_softmax(targets_u, 1) - F.log_softmax(logits_u, dim=1)), 1))

    if args.entropy_cost >0:
        entropy_loss = - torch.mean(torch.sum(torch.mul(F.softmax(logits_u,dim=1), F.log_softmax(logits_u,dim=1)),dim=1))
    else:
        entropy_loss = 0
    return class_loss + args.consistency_weight * consistency_loss + args.entropy_cost * entropy_loss, class_loss, consistency_loss


def get_u_label(model, loader,all_labels):
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _, index) in enumerate(loader):
            inputs = inputs.cuda()

            # compute output
            logits = model(inputs)
            all_labels[index] = logits.cpu().numpy()

    return all_labels


def scheduler(epoch,totals=None,start=0.0,end=1.0):
    if totals is None:
        totals = args.epochs
    step_ratio = epoch/totals
    if args.scheduler == 'linear':
        coeff = step_ratio
    elif args.scheduler == 'exp':
        coeff = np.exp((step_ratio - 1) * 5)
    elif args.scheduler == 'log':
        coeff = 1 - np.exp((-step_ratio) * 5)
    else:
        return 1.0
    return coeff * (end - start) +start


def get_unsup_size(epoch):
    size = int(min(args.mixup_size,args.unsup_ratio)*args.batch_size*scheduler(epoch))
    return size

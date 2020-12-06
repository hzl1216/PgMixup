from train_tool import *
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from models.wideresnet import WideResNet
from models.ema import ModelEMA
from dataset.cifar10 import get_cifar10
from dataset.svhn import get_svhn
import os
import random
from set_args import create_parser
from dataset.data_augment import get_data_augment
def main(dataset):
    print('start train %s '%dataset)
    def create_model():
        model = WideResNet(num_classes=10)
        model = model.cuda()
        return model

    transform_aug, transform_normal, transform_val = get_data_augment(dataset)
    if args.autoaugment:
        transform_1, transform_2 = transform_aug, transform_normal
    else:
        transform_1, transform_2 = transform_normal, transform_normal
    if dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set, train_unlabeled_set2, test_set = get_cifar10('./data', args.n_labeled,
                                                                                    transform_1=transform_1,
                                                                                    transform_2=transform_2,
                                                                                    transform_val=transform_val)
    if dataset == 'svhn':
        train_labeled_set, train_unlabeled_set, train_unlabeled_set2, val_set, test_set = get_svhn('./data', args.n_labeled, transform_1=transform_1, transform_2=transform_2, transform_val=transform_val)
    train_labeled_loader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          drop_last=True)
    train_unlabeled_loader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size*args.unsup_ratio, shuffle=True,
                                            num_workers=0, drop_last=True)
    train_unlabeled_loader2 = data.DataLoader(train_unlabeled_set2, batch_size=args.batch_size*args.unsup_ratio, shuffle=False,
                                            num_workers=0)

    test_loader = data.DataLoader(test_set, batch_size=args.batch_size*args.unsup_ratio, shuffle=False, num_workers=0)
    model = create_model()
    ema_model = ModelEMA(args, model, args.ema_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=True)

    totals = args.epochs*args.epoch_iteration
    warmup_step = args.warmup_step*args.epoch_iteration
    scheduler = WarmupCosineSchedule(optimizer, warmup_step, totals)
    all_labels = torch.zeros([len(train_unlabeled_set), 10])
    # optionally resume from a checkpoint
    title = dataset
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Evaluating the  model:")

        test_loss, test_acc = test(test_loader, ema_model.ema, criterion)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        logger = Logger(os.path.join(args.out_path, '%s_log_%d.txt'%(dataset,args.n_labeled)), title=title, resume=True)
        logger.append([args.start_epoch, 0, 0, test_loss, test_acc])
    else:
        logger = Logger(os.path.join(args.out_path, '%s_log_%d.txt'%(dataset,args.n_labeled)), title=title)
        logger.set_names(['epoch', 'Train_class_loss',  'Train_consistency_loss',  'Test_Loss', 'Test_Acc.'])

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch

        class_loss, cons_loss = train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model, optimizer, all_labels, epoch, scheduler)
        all_labels = get_u_label(ema_model.ema, train_unlabeled_loader2, all_labels)
        print("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            print("Evaluating the  model:")
            test_loss, test_acc = test(test_loader, model, criterion)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, test_loss, test_acc])

            print("Evaluating the EMA model:")
            ema_test_loss, ema_test_acc = test(test_loader, ema_model.ema, criterion)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss,ema_test_loss, ema_test_acc])

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint(
                '%s_%d'%(dataset, args.n_labeled),
                {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, 'checkpoint_path', epoch + 1)


def setup_seed(seed):
    random.seed(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    dirs = [ 'result', 'data', 'checkpoint_path']
    for path in dirs:
        if os.path.exists(path) is False:
            os.makedirs(path) 
    args = create_parser()
    set_args(args)
    setup_seed(args.seed)
    main(args.dataset)


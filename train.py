from train_tool import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from util.net import WideResNet
from util.cifar10 import get_cifar10
from util.svhn import get_svhn
import os
from set_args import create_parser
from util.data_augment import get_data_augment
def main(dataset):
    print('start train %s '%dataset)
    def create_model(ema=False):
        print("=> creating {ema}model ".format(
            ema='EMA ' if ema else ''))
        model = WideResNet(num_classes=10)
        model = nn.DataParallel(model).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    transform_aug, transform_normal, transform_val = get_data_augment(dataset)
    if dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set, train_unlabeled_set2,val_set, test_set = get_cifar10('./data', args.n_labeled, args.val_size,
                                                                                    transform_normal=transform_normal,
                                                                                    transform_aug=transform_aug,
                                                                                    transform_val=transform_val)
    if dataset == 'svhn':
        train_labeled_set, train_unlabeled_set, train_unlabeled_set2, val_set, test_set = get_svhn('./data',
                                                                                                      args.n_labeled,
                                                                                                      transform_normal=transform_normal,
                                                                                                      transform_aug=transform_aug,
                                                                                                      transform_val=transform_val)
    train_labeled_loader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          drop_last=True)
    train_unlabeled_loader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size*args.unsup_ratio, shuffle=True,
                                            num_workers=0, drop_last=True)
    train_unlabeled_loader2 = data.DataLoader(train_unlabeled_set2, batch_size=args.batch_size*args.unsup_ratio, shuffle=False,
                                            num_workers=0)

    if args.val_size>0:
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size*args.unsup_ratio, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size*args.unsup_ratio, shuffle=False, num_workers=0)
    model = create_model()
    ema_model = create_model(ema=True)
    tmp_model= create_model(ema=True)

    criterion = nn.CrossEntropyLoss().cuda()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    ema_optimizer = WeightEMA(model, ema_model, tmp_model, alpha=args.ema_decay)
    cudnn.benchmark = True
    if args.warmup_step>0:
        totals = args.epochs*args.epoch_iteration
        warmup_step = args.warmup_step*args.epoch_iteration
        scheduler =  WarmupCosineSchedule(optimizer,warmup_step,totals)
    else:
        scheduler = None
    all_labels = np.zeros([len(train_unlabeled_set), 10])
    # optionally resume from a checkpoint
    title = dataset
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Evaluating the  model:")
        if args.val_size>0:
            val_loss, val_acc = validate(val_loader, model, criterion,args.start_epoch)
        else:
            val_loss, val_acc = 0, 0
        test_loss, test_acc = validate(test_loader, model, criterion, args.start_epoch)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        logger = Logger(os.path.join(args.out_path, '%s_log_%d.txt'%(dataset,args.n_labeled)), title=title, resume=True)
        logger.append([args.start_epoch, 0, 0, val_loss, val_acc,test_loss, test_acc])
    else:
        logger = Logger(os.path.join(args.out_path, '%s_log_%d.txt'%(dataset,args.n_labeled)), title=title)
        logger.set_names(['epoch', 'Train_class_loss',  'Train_consistency_loss', 'Val_Loss', 'Val_Acc.', 'Test_Loss', 'Test_Acc.'])

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        class_loss, cons_loss,all_labels = train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model,optimizer, ema_optimizer,all_labels, epoch, scheduler)
        if epoch >= args.ema_stage:
            all_labels = get_u_label(ema_model, train_unlabeled_loader2,all_labels)

        print("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            print("Evaluating the  model:")
            if args.val_size>0:
                val_loss, val_acc = validate(val_loader, model, criterion,epoch)
            else:
                val_loss, val_acc = 0, 0
            test_loss, test_acc = validate(test_loader, model, criterion, epoch)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, val_loss, val_acc,test_loss, test_acc])

            print("Evaluating the EMA model:")
            if args.val_size > 0:
                ema_val_loss, ema_val_acc = validate(val_loader, ema_model, criterion,epoch)
            else:
                ema_val_loss, ema_val_acc = 0, 0
            ema_test_loss, ema_test_acc = validate(test_loader, ema_model, criterion, epoch)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, ema_val_loss, ema_val_acc,ema_test_loss, ema_test_acc])

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint(
                '%s_%d'%(dataset, args.n_labeled),
                {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'checkpoint_path', epoch + 1)

if __name__ == '__main__':
    args = create_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        np.random.seed(args.seed)
    set_args(args)
    main(args.dataset)


import os
import shutil
import argparse
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from easydict import EasyDict

from utils.misc import BlackHole, get_logger, get_new_log_dir, inf_iterator, load_config, seed_all, Counter
from utils.train import ValidationLossTape, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses
from utils.vc import get_version, has_changes
from utils.transform import get_transform
from datasets.shapenet import ShapeNetDataset
from models.partseg import get_model


def calculate_shape_IoU(pred_np, seg_np, num_seg_cls):
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):

        part_ious = []
        for part in range(num_seg_cls):
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('category', type=str, choices=['Airplane', 'Bag', 'Cap', 'Car', 'Chair',
                                 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 
                                 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_seg')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_lr', type=float, default=None)
    args = parser.parse_args()

    # Version control
    branch, version = get_version()
    version_short = '%s-%s' % (branch, version[:7])
    if has_changes() and not args.debug:
        exit()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix='%s(%s)[%s]' % (config_name, args.category, version_short), tag=args.tag)
            shutil.copytree('./models', os.path.join(log_dir, 'models'))
            shutil.copytree('./modules', os.path.join(log_dir, 'modules'))
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        if not os.path.exists( os.path.join(log_dir, os.path.basename(args.config)) ):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)


    # Data
    logger.info('Loading datasets...')
    ## Datasets
    train_set = ShapeNetDataset(config.data.root, npoint=config.data.npoint, split='train', transform=get_transform(config.data.train_transform), class_choice=args.category)
    test_set = ShapeNetDataset(config.data.root, npoint=config.data.npoint, split='test', transform=get_transform(config.data.test_transform), class_choice=args.category)
    ## Dataloaders
    loader_train = DataLoader(train_set, batch_size=config.data.batch_size, shuffle=True, drop_last=True, num_workers=8)
    loader_test = DataLoader(test_set, batch_size=config.data.batch_size, shuffle=False, num_workers=8)
    logger.info('Training data: %d' % len(train_set))
    logger.info('Test data: %d' % len(test_set))

    # Model
    logger.info('Building model...')
    model = get_model(config.model, train_set.seg_num_classes).to(args.device)

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_global = Counter(1)
    epoch_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_global.set(ckpt['iteration'] + 1)
        epoch_ckpt = int(os.path.basename(args.resume).split('.')[0])
        epoch_first = epoch_ckpt + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.resume_lr is not None:
            optimizer.param_groups[0]['lr'] = args.resume_lr
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    def train(epoch):
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for i, batch in enumerate(tqdm(loader_train, desc='Train #%d' % epoch, dynamic_ncols=True)):
            batch = recursive_to(batch, args.device)
            
            # Set states
            model.train()

            # Forward pass
            point = batch['point']              # (B, N, 3)
            cat = batch['cls'].flatten().long() # (B, )
            target = batch['seg'].long()        # (B, N)
            loss, pred = model.get_loss(point, target, cat)  # (B, N, num_seg_cls)

            # Backward pass
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            log_losses(EasyDict({'overall': loss}), it_global.now, 'train', BlackHole(), writer, others={
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],  
            })
            it_global.step()

            # Metrics
            pred_seg = pred.max(dim=-1)[1]  # (B, N)
            seg_np = target.cpu().numpy()               # (batch_size, num_points)
            pred_np = pred_seg.detach().cpu().numpy()   # (batch_size, num_points)
            # print('seg_np ', seg_np)
            # print('pred_np', pred_np)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(cat.cpu().numpy().reshape(-1))

        scheduler.step()

        if epoch % config.train.val_freq == 0:
            train_true_cls = np.concatenate(train_true_cls)
            train_pred_cls = np.concatenate(train_pred_cls)
            train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
            train_true_seg = np.concatenate(train_true_seg, axis=0)
            train_pred_seg = np.concatenate(train_pred_seg, axis=0)
            train_label_seg = np.concatenate(train_label_seg)
            # print(train_pred_seg)
            # print(train_true_seg)
            train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_set.seg_num_classes)
            
            outstr = 'Train %d: Instance Accuracy: %.2f, Balanced Accuracy: %.2f, IoU: %.2f' % (
                epoch, 
                train_acc * 100,
                avg_per_class_acc * 100,
                np.mean(train_ious) * 100,
            )
            logger.info(outstr)

            writer.add_scalar('train/acc_instance', train_acc, epoch)
            writer.add_scalar('train/acc_balanced', avg_per_class_acc, epoch)
            writer.add_scalar('train/IoU', np.mean(train_ious), epoch)

    def test(epoch):
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        with torch.no_grad():
            for batch in tqdm(loader_test, desc='Test #%d' % epoch, dynamic_ncols=True):
                batch = recursive_to(batch, args.device)
                
                # Set states
                model.eval()

                # Forward pass
                point = batch['point']              # (B, N, 3)
                cat = batch['cls'].flatten().long() # (B, )
                target = batch['seg'].long()        # (B, N)
                loss, pred = model.get_loss(point, target, cat)  # (B, N, num_seg_cls)

                # Metrics
                pred_seg = pred.max(dim=-1)[1]  # (B, N)
                seg_np = target.cpu().numpy()               # (batch_size, num_points)
                pred_np = pred_seg.detach().cpu().numpy()   # (batch_size, num_points)
                train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
                train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
                train_true_seg.append(seg_np)
                train_pred_seg.append(pred_np)
                train_label_seg.append(cat.cpu().numpy().reshape(-1))

            train_true_cls = np.concatenate(train_true_cls)
            train_pred_cls = np.concatenate(train_pred_cls)
            train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
            train_true_seg = np.concatenate(train_true_seg, axis=0)
            train_pred_seg = np.concatenate(train_pred_seg, axis=0)
            train_label_seg = np.concatenate(train_label_seg)
            train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_set.seg_num_classes)
            
            outstr = 'Test %d:  Instance Accuracy: %.2f, Balanced Accuracy: %.2f, IoU: %.2f' % (
                epoch, 
                train_acc * 100,
                avg_per_class_acc * 100,
                np.mean(train_ious) * 100,
            )
            logger.info(outstr)

            writer.add_scalar('test/acc_instance', train_acc, epoch)
            writer.add_scalar('test/acc_balanced', avg_per_class_acc, epoch)
            writer.add_scalar('test/IoU', np.mean(train_ious), epoch)

            return train_acc, avg_per_class_acc

    try:
        for epoch in range(epoch_first, config.train.max_epochs+1):
            train(epoch)
            if epoch % config.train.val_freq == 0:
                acc, acc_by_cat = test(epoch)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % epoch)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it_global.now,
                        'acc': acc,
                        'acc_by_cat': acc_by_cat
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

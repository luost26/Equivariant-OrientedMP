import os
import shutil
import argparse
import numpy as np
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
from datasets.modelnet import ModelNetDataset
from models.cls import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
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
            log_dir = get_new_log_dir(args.logdir, prefix='%s[%s]' % (config_name, version_short), tag=args.tag)
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
    train_set = ModelNetDataset(config.data.root, npoint=config.data.npoint, split='train', transform=get_transform(config.data.train_transform))
    test_set = ModelNetDataset(config.data.root, npoint=config.data.npoint, split='test', transform=get_transform(config.data.test_transform))
    ## Dataloaders
    loader_train = DataLoader(train_set, batch_size=config.data.batch_size, shuffle=True, drop_last=True, num_workers=8)
    loader_test = DataLoader(test_set, batch_size=config.data.batch_size, shuffle=False, num_workers=8)
    logger.info('Training data: %d' % len(train_set))
    logger.info('Test data: %d' % len(test_set))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)

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
        acc_all = []
        for i, batch in enumerate(tqdm(loader_train, desc='Train #%d' % epoch, dynamic_ncols=True)):
            batch = recursive_to(batch, args.device)
            
            # Set states
            model.train()

            # Forward pass
            point = batch['point']
            target = batch['cls'].flatten().long()
            loss, logp_pred = model.get_loss(point, target, return_result=True)
            pred_choice = logp_pred.data.max(1)[1]
            correct = pred_choice.eq(target).cpu().sum()
            acc_all.append(correct.item() / float(point.size(0)))

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

        scheduler.step()
        acc_avg = np.mean(acc_all)
        logger.info('Train Accuracy: %.2f' % (acc_avg * 100))
    

    def test(epoch):
        correct_all = []
        correct_by_cat = [[] for _ in range(config.model.num_classes)]
        with torch.no_grad():
            for batch in tqdm(loader_test, desc='Test #%d' % epoch, dynamic_ncols=True):
                batch = recursive_to(batch, args.device)
                
                # Set states
                model.eval()

                # Forward pass
                logp_pred = model(batch['point'])

                # Accuracy
                gts_cat = batch['cls'].view([batch['cls'].size(0)]).cpu().long()
                pred_cat = logp_pred.data.max(1)[1].cpu()
                for cat in np.unique(gts_cat.tolist()):
                    cat_correct = (pred_cat[gts_cat == cat] == gts_cat[gts_cat == cat]).long().tolist()
                    correct_all.extend(cat_correct)
                    correct_by_cat[cat].extend(cat_correct)
        
        acc = np.mean(correct_all)
        acc_by_cat = [np.mean(l) for l in correct_by_cat]

        logger.info('Test Instance Accuracy %.2f, Class Accuracy %.2f' % (acc*100, np.mean(acc_by_cat)*100))
        writer.add_scalar('test/acc_instance', acc, epoch)
        writer.add_scalar('test/acc_class', np.mean(acc_by_cat), epoch)
        return acc, acc_by_cat

    try:
        for epoch in range(epoch_first, config.train.max_epochs+1):
            train(epoch)
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

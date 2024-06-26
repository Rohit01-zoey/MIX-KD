"""
Framework of Mix-KD, WACV2023
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from helper.util import adjust_learning_rate
from distiller_zoo import DistillKL, MIXSTDLoss
from helper.loops_mixstd import train_distill as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default='./checkpoints/resnet56/best.pth', help='teacher model snapshot')
    # resnet32x4_vanilla, ResNet50_vanilla, resnet56_vanilla, resnet110_vanilla, vgg13_vanilla, wrn_40_2_vanilla

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # distillation
    parser.add_argument('--distill', type=str, default='mixstd', choices=['mixstd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=0.1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('--pmixup', default=True, help='True=use PMU, False=use FMU')
    parser.add_argument('--partmixup', type=float, default=0.90, help='amount of PMU, e.g., 0.5=50%, 0.9=10%')
    parser.add_argument('--scale_T', type=float, default=1, help='temperature for mixup distillation')
    parser.add_argument('--beta_a', type=float, default=1.0, help='beta distribution, 0.2, 0.4, 1.0')
    opt = parser.parse_args()



    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
        #opt.learning_rate = 0.001

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S={}_T={}_{}_{}_r={}_a={}_b={}_pmix={}_{}_betalpha={}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.pmixup, opt.partmixup, opt.beta_a, opt.trial)
    
    
    # Create a logger
    logger = logging.getLogger(__name__)

    # Set the level of this logger. DEBUG is the lowest level. 
    # Therefore, all messages will be logged.
    logger.setLevel(logging.DEBUG)
    
    base_dir = './logs/S={}_T={}'.format(opt.model_s, opt.model_t)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    RUN = 1
    FILE_NAME = '{}_{}_r={}_a={}_b={}_pmix={}_{}_betalpha={}_{}_[R{}].txt'.format(opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.pmixup, opt.partmixup, opt.beta_a, opt.trial, RUN)
    LOG_FILE_PATH = os.path.join(base_dir, FILE_NAME)
    while os.path.exists(LOG_FILE_PATH):
        RUN += 1
        FILE_NAME = '{}_{}_r={}_a={}_b={}_pmix={}_{}_betalpha={}_{}_[R{}].txt'.format(opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.pmixup, opt.partmixup, opt.beta_a, opt.trial, RUN)
        LOG_FILE_PATH = os.path.join(base_dir, FILE_NAME)
        
    # Create a file handler for outputting log messages to a file
    fh = logging.FileHandler(LOG_FILE_PATH)
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('[%(asctime)s - %(name)s] %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)
    opt.logger = logger

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path)['state_dict'])
        print('==> load done (cuda)')
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu')['state'])
        print('==> load done (cpu)')

    return model


def main():
    best_acc = 0

    opt = parse_option()


    print('#####     Learning MIXSTD    #####')
    print('Teacher Model     :          {}'.format(opt.path_t))
    print('Student Model     :          {}'.format(opt.model_s))
    print('Distill method    :          {}'.format(opt.distill))
    print('Weight gamma      :          {}'.format(opt.gamma))
    print('Weight alpha      :          {}'.format(opt.alpha))
    print('Weight beta       :          {}'.format(opt.beta))
    print('')
    print('using pmixup?      :          {}'.format(opt.pmixup))
    print('partmixup          :          {}'.format(opt.partmixup))
    print('betalpha           :          {}'.format(opt.beta_a))
    print('learning_rate      :          {}'.format(opt.learning_rate))
    print('')



    # tensorboard logger
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger = opt.logger

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
        
    elif opt.distill == 'mixstd':
        criterion_kd = MIXSTDLoss(opt)

    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)
    logger.info('teacher_acc: %s', teacher_acc.item())

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.info('Epoch: %s | train_acc: %s | train_loss:%s', epoch, train_acc.item(), train_loss)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.info('Epoch: %s | test_acc: %s| test_acc_top5: %s | test_loss:%s', epoch, test_acc.item(), tect_acc_top5.item(), test_loss)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    logger.info('Best Test accuracy: %s', best_acc.item())
    print('best accuracy:', best_acc)
    print('')


    print('#####     Learning MIXSTD    #####')
    print('Teacher Model     :          {}'.format(opt.path_t))
    print('Student Model     :          {}'.format(opt.model_s))
    print('Distill method    :          {}'.format(opt.distill))
    print('Weight gamma      :          {}'.format(opt.gamma))
    print('Weight alpha      :          {}'.format(opt.alpha))
    print('Weight beta       :          {}'.format(opt.beta))
    print('')
    print('using pmixup?      :          {}'.format(opt.pmixup))
    print('partmixup          :          {}'.format(opt.partmixup))
    print('betalpha           :          {}'.format(opt.beta_a))
    print('learning_rate      :          {}'.format(opt.learning_rate))
    print('')

    
    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()

from __future__ import print_function
import os
import argparse
from models import *
from attacker import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import shutil
import logging
import os
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)
from attacker import *
from FGSM_SDI_loss import adv_FGSM_loss
logger = logging.getLogger('logger')
from torch import autograd
import numpy as np
def get_args():
    parser = argparse.ArgumentParser('FGSM-SDI')

    parser.add_argument('--batch_size', default=128, type=int)
    #parser.add_argument('--model_num_labels', default=10, type=int)

    parser.add_argument('--target_model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_att', type=float, default=0.001,
                        help='attacker learning rate')
    parser.add_argument('--opt_att', type=str, default='Adam', \
                        help='attacker optimizer')
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--factor', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--beta', default=1.0, type=float,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--data-dir', default='D:\\Auto_AT\\data\\cifar-data', type=str)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--att_num', default=20, type=int, \
                        help='number of iterations for attacker update')
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    arguments = parser.parse_args()
    return arguments
args = get_args()
output_path = os.path.join(args.out_dir, 'epochs_' + str(args.epochs))
output_path = os.path.join(output_path, 'attacker_' + args.attacker)
output_path = os.path.join(output_path, 'factor_' + str(args.factor))
output_path = os.path.join(output_path, 'target_model_' + args.target_model)
output_path = os.path.join(output_path, 'opt_att_' + args.opt_att)
output_path = os.path.join(output_path, 'lr_att_' + str(args.lr_att))
output_path = os.path.join(output_path, 'lr_max_' + str(args.lr_max))
output_path = os.path.join(output_path, 'alpha_' + str(args.alpha))
output_path = os.path.join(output_path, 'att_num_' + str(args.att_num))

#output_path = os.path.join(output_path, 'feature_lambda_' + str(args.feature_lambda))
from tensorboardX import SummaryWriter
summary_log_dir=os.path.join(output_path,"Attack")
if not os.path.exists(summary_log_dir):
    os.makedirs(summary_log_dir)
summary_writer = SummaryWriter(log_dir=summary_log_dir, comment="good_makeatari")
if not os.path.exists(output_path):
    os.makedirs(output_path)
logfile = os.path.join(output_path, 'output.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(output_path, 'output.log'))
logger.info(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std


if args.target_model == "VGG":
    target_model = VGG('VGG19')
elif args.target_model == "ResNet18":
    target_model = ResNet18()
elif args.target_model == "PreActResNest18":
    target_model = PreActResNet18()
elif args.target_model == "WideResNet":
    target_model = WideResNet()
target_model = target_model.cuda()


attacker = One_Layer_Attacker(eps=epsilon, input_channel=6).cuda()

train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

target_model_optimizer=torch.optim.SGD(target_model.parameters(),lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)

if args.opt_att == 'SGD':
    optimizer_att = torch.optim.SGD(attacker.parameters(), lr=args.lr_att, momentum=args.momentum,
                              weight_decay=args.weight_decay)
else:
    optimizer_att = torch.optim.Adam(attacker.parameters(), lr=args.lr_att, weight_decay=args.weight_decay)
lr_steps = args.epochs * len(train_loader)
if args.lr_schedule == 'cyclic':
    target_model_scheduler = torch.optim.lr_scheduler.CyclicLR(target_model_optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
                                                  step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.lr_schedule == 'multistep':
    target_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(target_model_optimizer, milestones=[lr_steps * 100/110, lr_steps * 105/110],
                                                     gamma=0.1)


attacker_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_att,
                                                      milestones=[lr_steps * 100/110, lr_steps * 105/ 110],
                                                      gamma=0.1)
import numpy as np
from torch.autograd import Variable
def _label_smoothing(label, factor):
    one_hot = np.eye(10)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))

    return result
def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

def train(args, model, attacker, train_loader, optimizer, optimizer_att, epoch):
    epoch_time = 0
    train_loss = 0
    train_acc = 0
    train_n = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time=time.time()
        data, target = data.cuda(), target.cuda()
        label_smoothing = Variable(torch.tensor(_label_smoothing(target, args.factor)).cuda())
        data.requires_grad_()
        with torch.enable_grad():
            #loss_natural = LabelSmoothLoss(model(data), label_smoothing.float())

            loss_natural = F.cross_entropy(model(data), target)
        grad = torch.autograd.grad(loss_natural, [data])[0]




        # for _ in range(args.att_iter):
        if batch_idx %args.att_num==0:
            optimizer_att.zero_grad()

            attacker.zero_grad()
            model.zero_grad()
            loss_adv, output = adv_FGSM_loss(model=model,
                                             grad=grad,
                                             factor=args.factor,
                                             attacker=attacker,
                                             x_natural=data,
                                             y=target,
                                             optimizer=optimizer,
                                             optimizer_att=optimizer_att,
                                             epsilon=epsilon,
                                             alpha=alpha,
                                             for_attacker=1)
            loss_adv=-loss_adv

            loss_adv.backward()

            optimizer_att.step()
            attacker_scheduler.step()
            print(loss_adv)
        target_model_optimizer.zero_grad()
        model.zero_grad()
        attacker.zero_grad()
        # calculate robust loss
        loss, output = adv_FGSM_loss(model=model,
                                       grad=grad,
                                       factor=args.factor,
                                       attacker=attacker,
                                       x_natural=data,
                                       y=target,
                                       optimizer=optimizer,
                                       optimizer_att=optimizer_att,
                                       epsilon=epsilon,
                                       alpha=alpha,
                                       for_attacker=0)
        loss.backward()

        target_model_optimizer.step()
        target_model_scheduler.step()
        train_loss += loss.item() * target.size(0)
        train_acc += (output.max(1)[1] == target).sum().item()
        train_n += target.size(0)

        print(loss)
        batch_end_time=time.time()
        epoch_time += batch_end_time-batch_start_time
    lr = target_model_scheduler.get_lr()[0]
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                epoch, epoch_time, lr, train_loss / train_n, train_acc / train_n)
    print(epoch, epoch_time, lr, train_loss / train_n, train_acc / train_n)

def main():
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    #logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(1, args.epochs + 1):
        train(args, target_model, attacker,  train_loader, target_model_optimizer, optimizer_att, epoch)
    # evaluation on natural examples
        logger.info('==> Building model..')
        if args.target_model == "VGG":
            model_test = VGG('VGG19').cuda()
        elif args.target_model == "ResNet18":
            model_test = ResNet18().cuda()
        elif args.target_model == "PreActResNest18":
            model_test = PreActResNet18().cuda()
        elif args.target_model == "WideResNet":
            model_test = WideResNet().cuda()

        model_test.load_state_dict(target_model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        if best_result<=pgd_acc:
            best_result=pgd_acc
            torch.save(target_model.state_dict(), os.path.join(output_path, 'best_model.pth'))
        torch.save(target_model.state_dict(), os.path.join(output_path, 'final_model.pth'))
    # model_test = PreActResNet18().cuda()
    # model_test.load_state_dict(target_model.state_dict())
    # model_test.float()
    # model_test.eval()
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    print(epoch_clean_list)
    print(epoch_pgd_list)


main()



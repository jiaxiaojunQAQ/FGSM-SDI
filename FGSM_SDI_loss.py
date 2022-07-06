import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)
def l2_norm(x):
    return squared_l2_norm(x).sqrt()


import numpy as np
def _label_smoothing(label, factor):
    one_hot = np.eye(10)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))

    return result
def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


def adv_FGSM_loss(model,
                grad,
                factor,
                attacker,
                x_natural,
                y,
                optimizer,
                optimizer_att,
                epsilon =(8 / 255.) / std,
                alpha=(8 / 255.) / std,
                for_attacker = 0):
    if for_attacker == 0:
        model.train()
        attacker.eval()
    else:
        #model.eval()
        attacker.train()
    label_smoothing = Variable(torch.tensor(_label_smoothing(y, factor)).cuda())
    # x_natural.requires_grad_()
    # with torch.enable_grad():
    #     loss_natural = F.cross_entropy(model(x_natural), y)
    # grad = torch.autograd.grad(loss_natural, [x_natural])[0]
    advinput = torch.cat([x_natural, 1.0 * (torch.sign(grad))], 1).detach()

    # generate adversarial example
    perturbation = attacker(advinput)

    x_adv = x_natural + perturbation
    #x_adv = clamp(x_adv, lower_limit, upper_limit)

    x_adv.requires_grad_()
    # model.eval()
    with torch.enable_grad():
        #loss_adv = LabelSmoothLoss(model(x_adv), label_smoothing.float())

        loss_adv = F.cross_entropy(model(x_adv), y)
        grad_adv = torch.autograd.grad(loss_adv, [x_adv])[0]
        perturbation_1= clamp( alpha* torch.sign(grad_adv), -epsilon, epsilon)

    perturbation_total = perturbation + perturbation_1
    perturbation_total = clamp(perturbation_total, -epsilon, epsilon)

    x_adv_final = x_natural + perturbation_total

    # optimizer.zero_grad()
    # optimizer_att.zero_grad()
    output=model(x_adv_final)
    loss_robust = LabelSmoothLoss(output,label_smoothing.float())

    loss = loss_robust
    return loss,output



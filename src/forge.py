from pyexpat import features
import imageio
import os
import shutil
import sys
from scipy.sparse import data
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append('.')  # noqa: E402
from model import RACNN
from plant_loader import get_plant_loader
from pretrain_apn import random_sample, log, clean, save_img, build_gif


def avg(x): return sum(x)/len(x)


def train(net, dataloader, optimizer, epoch, _type):
    assert _type in ['apn', 'backbone']
    losses = 0
    net.mode(_type), log(f' :: Switch to {_type}')  # switch loss type
    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss = net.echo(inputs, targets, optimizer)
        losses += loss

        if step % 20 == 0 and step != 0:
            avg_loss = losses/20
            log(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
            losses = 0

    return avg_loss


def test(net, dataloader):
    correct = [0, 0, 0]
    cnt = 0
    avg_accuracy = 0

    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        cnt += inputs.size(0)
        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            for idx, logits in enumerate(outputs):
                logits = torch.sigmoid(logits)
                logits[logits >= 0.5] = 1
                logits[logits < 0.5] = 0
                # correct_summary[f'clsf-{idx}']['top-1'] +=  torch.all(torch.eq(logits, labels),  dim=1).sum()  # top-1
                correct[idx] += torch.all(torch.eq(logits,
                                          labels),  dim=1).sum()
                # correct_summary[f'clsf-{idx}']['top-5'] += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-5
    for id, value in enumerate(correct):
        correct[id] = (value/cnt).detach().cpu()
        avg_accuracy += (correct[id]).detach().cpu()

    return avg_accuracy/3, correct


def run(pretrained_model, runPath='runs/exp1'):
    accuracy = 0
    log(f' :: Start training with {pretrained_model}')
    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    cudnn.benchmark = True

    # ognuna delle 3 cnn del modello parte con i valori della cnn pre addestrata e poi ognuna si specializza
    # con i propri parametri
    cls1_params = list(net.bf1.parameters()) + \
        list(net.classifier1.parameters())
    cls2_params = list(net.bf2.parameters()) + \
        list(net.classifier2.parameters())
    cls3_params = list(net.bf3.parameters()) + \
        list(net.classifier3.parameters())
    apn1_params = list(net.apn1.parameters())
    apn2_params = list(net.apn2.parameters())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    cls_opt = [
        optim.SGD(cls1_params, lr=0.001, momentum=0.9),
        optim.SGD(cls2_params, lr=0.001, momentum=0.9),
        optim.SGD(cls3_params, lr=0.001, momentum=0.9),
    ]
    apn_opt = [
        optim.SGD(apn1_params, lr=1e-6),
        optim.SGD(apn2_params, lr=1e-6),
    ]

    data_set = get_plant_loader()
    trainloader = torch.utils.data.DataLoader(
        data_set["train"], batch_size=10, shuffle=True)
    validationloader = torch.utils.data.DataLoader(
        data_set["validation"], batch_size=8, shuffle=False)
    sample = random_sample(validationloader)

    n_train_steps = len(trainloader)
    n_valid_steps = len(validationloader)
    # print(n_train_steps, n_valid_steps)

    # print(cls_opt[0].state_dict())
    # sys.exit()

    writer = SummaryWriter(runPath)

    it = iter(validationloader)
    images, labels = it.next()
    writer.add_graph(net, images.cuda())
    writer.close()

    # 15
    for epoch in range(40):

        cls_loss = train(net, trainloader, cls_opt, epoch, 'backbone')
        # save to tensorboard
        writer.add_scalar('racnn cls loss', cls_loss, epoch*n_train_steps)

        log(' :: Testing on validation set ...')
        temp_accuracy, valid_corrects = test(net, validationloader)
        # save to tensorboard
        writer.add_scalar('racnn validation accuracy',
                          temp_accuracy, epoch*n_valid_steps)
        writer.add_scalar('racnn cls1 validation accuracy',
                          valid_corrects[0], epoch*n_valid_steps)
        writer.add_scalar('racnn cls2 validation accuracy',
                          valid_corrects[1], epoch*n_valid_steps)
        writer.add_scalar('racnn cls3 validation accuracy',
                          valid_corrects[2], epoch*n_valid_steps)

        print("validation accuracies:", valid_corrects)

        log(' :: Testing on training set ...')
        train_accuracy, train_corrects = test(net, trainloader)
        # save to tensorboard
        writer.add_scalar('racnn training accuracy',
                          train_accuracy, epoch*n_train_steps)
        writer.add_scalar('racnn cls1 training accuracy',
                          train_corrects[0], epoch*n_train_steps)
        writer.add_scalar('racnn cls2 training accuracy',
                          train_corrects[1], epoch*n_train_steps)
        writer.add_scalar('racnn cls3 training accuracy',
                          train_corrects[2], epoch*n_train_steps)

        rank_loss = train(net, trainloader, apn_opt, epoch, 'apn')
        # save to tensorboard
        writer.add_scalar('racnn rank loss', rank_loss, epoch*n_train_steps)

        # visualize cropped inputs
        _, _, _, resized = net(sample.unsqueeze(0))
        x1, x2 = resized[0].data, resized[1].data
        save_img(x1, path=f'build/.cache/epoch_{epoch}@2x.jpg')
        save_img(x2, path=f'build/.cache/epoch_{epoch}@4x.jpg')

        if temp_accuracy > accuracy:
            accuracy = temp_accuracy
            torch.save(net.state_dict(), f'build/racnn_efficientNetB0.pt')
            log(f' :: Saved model dict as:\tbuild/racnn_efficientNetB0.pt')
            #torch.save(cls_opt.state_dict(), f'build/cls_optimizer.pt')
            #torch.save(apn_opt.state_dict(), f'build/apn_optimizer.pt')


if __name__ == "__main__":
    clean()
    path = 'runs'
    if not os.path.exists(path):
        os.makedirs(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # RACNN con backbone e apn pre addestrate
    run(pretrained_model='build/racnn_pretrained.pt', runPath=f'{path}/exp1')
    build_gif(pattern='@2x', gif_name='racnn_efficientNet')
    build_gif(pattern='@4x', gif_name='racnn_efficientNet')

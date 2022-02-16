import os
from torch import nn, optim
import torch.backends.cudnn as cudnn
import torch
from model import RACNN
from plant_loader import get_plant_loader
from pretrain_apn import clean, log
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')


class FusionCLS(nn.Module):
    def __init__(self, num_classes):
        super(FusionCLS, self).__init__()
        self.RACNN = RACNN(num_classes=num_classes).cuda()
        
        self.RACNN.eval()

        for parameters in self.RACNN.parameters():
            parameters.requires_grad = False

        self.scale_1_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*2, num_classes),
            nn.Sigmoid(),
        )
        self.scale_1_2_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*3, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits, _, _, _ = self.RACNN(x)
        ret1 = self.scale_1_2(torch.cat((logits[0], logits[1]), dim=1))
        ret2 = self.scale_1_2_3(
            torch.cat((logits[0], logits[1], logits[2]), dim=1))
        return [ret1, ret2]

    @staticmethod
    def task_loss(logits, targets):
        criterion1 = torch.nn.BCELoss()
        criterion2 = torch.nn.BCELoss()
        return criterion1(logits[0], targets), criterion2(logits[1], targets)

    def __echo_backbone(self, inputs, targets, optimizer_1_2, optimizer_1_2_3):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        out = self.forward(inputs)
        # optimizer_1_2.zero_grad()
        # optimizer_1_2_3.zero_grad()

        loss_1_2, loss_1_2_3 = self.task_loss(out, targets)

        optimizer_1_2.zero_grad()
        loss_1_2.backward()
        optimizer_1_2.step()

        optimizer_1_2_3.zero_grad()
        loss_1_2_3.backward()
        optimizer_1_2_3.step()

        return loss_1_2.item(), loss_1_2_3.item()

    def load_state(self, pretrained_model):
        self.RACNN.load_state_dict(torch.load(pretrained_model))

    def mode(self, mode_type):
        assert mode_type in ['backbone']
        if mode_type == 'backbone':
            self.echo = self.__echo_backbone
            self.train()


def train(net, dataloader, optimizer_1_2, optimizer_1_2_3, epoch):
    losses = [0, 0]
    net.mode('backbone')
    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss_1_2, loss_1_2_3 = net.echo(
            inputs, targets, optimizer_1_2, optimizer_1_2_3)
        losses[0] += loss_1_2
        losses[1] += loss_1_2_3
        if step % 20 == 0 and step != 0:
            avg_loss_1_2 = losses[0]/20
            avg_loss_1_2_3 = losses[1]/20
            log(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: avg loss_1_2 {avg_loss_1_2:.10f} loss_1_2_3 {avg_loss_1_2_3:.10f}')
            losses[0] = 0
            losses[1] = 0
    return avg_loss_1_2, avg_loss_1_2_3


def test(net, dataloader):
    correct = [0, 0]
    cnt = 0
    avg_accuracy = 0

    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        cnt += inputs.size(0)
        with torch.no_grad():
            outputs = net(inputs)
            for id, logits in enumerate(outputs):
                logits[logits >= 0.5] = 1
                logits[logits < 0.5] = 0
                correct[id] += torch.all(torch.eq(logits,
                                         labels),  dim=1).sum()
    for id, value in enumerate(correct):
        correct[id] = (value/cnt).detach().cpu()
        avg_accuracy += (correct[id]).detach().cpu()
    return avg_accuracy/2, correct


def run(pretrained_model):
    accuracy = 0
    log(f' :: Start training with {pretrained_model}')
    net = FusionCLS(num_classes=6).cuda()
    net.load_state(pretrained_model)
    cudnn.benchmark = True

    fusion_1_2_params = list(net.scale_1_2.parameters())
    fusion_1_2_3_params = list(net.scale_1_2_3.parameters())
    fusion_1_2_opt = optim.SGD(fusion_1_2_params, lr=0.001, momentum=0.9)
    fusion_1_2_3_opt = optim.SGD(fusion_1_2_3_params,  lr=0.001, momentum=0.9)

    data_set = get_plant_loader()
    trainloader = torch.utils.data.DataLoader(
        data_set["train"], batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(
        data_set["validation"], batch_size=32, shuffle=False)

    # 15
    for epoch in range(10):
        log(' :: Training ...')
        fusion_loss_1_2, fusion_loss_1_2_3 = train(
            net, trainloader, fusion_1_2_opt, fusion_1_2_3_opt, epoch)
        print(f' :: losses: {fusion_loss_1_2:.4f} {fusion_loss_1_2_3:.4f}')

        log(' :: Testing on validation set ...')
        tmp_accuracy, valid_accuracies = test(net, validationloader)
        print(" :: accuracies:", valid_accuracies)

        if tmp_accuracy > accuracy:
            print(f' :: {tmp_accuracy:.4f} > {accuracy:.4f}')
            accuracy = tmp_accuracy
            torch.save(net.state_dict(), f'build/fusion_efficientNetB0.pt')
            log(f' :: Saved model dict as:\tbuild/fusion_efficientNetB0.pt')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(pretrained_model='build/racnn_efficientNetB0.pt')

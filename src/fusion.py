from logging import logProcesses
import os
import time
from torch import nn, optim
import torch.backends.cudnn as cudnn
import torch
from forge import cleanlog
from model import RACNN
from plant_loader import get_plant_loader
from pretrain_apn import clean, log
from torch.autograd import Variable

class FusionCLS(nn.Module):
    def __init__(self, num_classes):
        super(FusionCLS, self).__init__()
        self.RACNN = RACNN(num_classes=num_classes).cuda()
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
        ret1 = self.scale_1_2([logits[0], logits[1]])
        ret2 = self.scale_1_2_3(logits)
        return ret1, ret2
    
    @staticmethod
    def task_loss(logits, targets):
        criterion1 = torch.nn.BCELoss()
        criterion2 = torch.nn.BCELoss()
        return criterion1(logits[0], targets), criterion2(logits[1], targets)

    def __echo_backbone(self, inputs, targets, optimizer_1_2, optimizer_1_2_3):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        out = self.forward(inputs)
        optimizer_1_2.zero_grad()
        optimizer_1_2_3.zero_grad()
        
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
    losses = [0,0]
    net.mode('backbone')
    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss_1_2, loss_1_2_3 = net.echo(inputs, targets, optimizer_1_2, optimizer_1_2_3)
        losses[0] += loss_1_2
        losses[1] += loss_1_2_3
        if step % 20 == 0 and step != 0:
            avg_loss_1_2 = losses[0]/20
            avg_loss_1_2_3 = losses[1]/20
            logProcesses(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: avg_loss_1_2 {avg_loss_1_2:.10f}')
            logProcesses(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: avg_loss_1_2_3 {avg_loss_1_2_3:.10f}')
            losses[0] = 0
            losses[1] = 0
    return avg_loss_1_2, avg_loss_1_2_3


def test(net, dataloader):
    correct = [0, 0, 0]
    cnt = 0
    avg_accuracy = 0

    return avg_accuracy/3, correct


def run(pretrained_model):
    accuracy = 0
    log(f' :: Start training with {pretrained_model}')
    net = FusionCLS(num_classes=6,)
    net.load_state(pretrained_model)
    cudnn.benchmark = True

    fusion_1_2_params = list(net.scale_1_2.parameters())
    fusion_1_2_3_params = list(net.scale_1_2_3.parameters())
    fusion_1_2_opt = optim.SGD(fusion_1_2_params, lr=0.001, momentum=0.9)
    fusion_1_2_3_opt = optim.SGD(fusion_1_2_3_params,  lr=0.001, momentum=0.9)

    data_set = get_plant_loader()
    trainloader = torch.utils.data.DataLoader(
        data_set["train"], batch_size=16, shuffle=True)
    validationloader = torch.utils.data.DataLoader(
        data_set["validation"], batch_size=16, shuffle=False)

    # 15
    for epoch in range(40):
        fusion_loss_1_2, fusion_loss_1_2_3 = train(net, trainloader, fusion_1_2_opt, fusion_1_2_3_opt, epoch)
        #log(' :: Testing on validation set ...')
        #temp_accuracy, valid_corrects = test(net, validationloader)

        #print("validation accuracies:", valid_corrects)

        #log(' :: Testing on training set ...')
        #train_accuracy, train_corrects = test(net, trainloader)

        # save model per 10 epoches
        #if temp_accuracy > accuracy:
        #    accuracy = temp_accuracy
        #    stamp = f'e{epoch}{int(time.time())}'
        #    torch.save(net.state_dict(), f'build/fusion_efficientNetB0.pt')
        #   log(f' :: Saved model dict as:\tbuild/fusion_efficientNetB0.pt')

if __name__ == "__main__":
    clean()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(pretrained_model='build/racnn_efficientNetB0.pt')

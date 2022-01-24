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
        self.RACNN = RACNN(num_classes=num_classes)
        self.scale_1_2 = nn.Sequential(
            nn.Linear(320*7*7*2, num_classes),
            nn.Sigmoid(),
        )
        self.scale_1_2_3 = nn.Sequential(
            nn.Linear(320*7*7*3, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, features, _, _ = self.RACNN(x)
        features_1_2 = torch.stack([
                features[0].view(-1, 320* 7*7),
                features[1].view(-1, 320* 7*7),
            ], dim=1)
        features_1_2_3 = torch.stack([
                features[0].view(-1, 320* 7*7),
                features[1].view(-1, 320* 7*7),
                features[2].view(-1, 320* 7*7),
            ], dim=1)    

        ret1 = self.scale_1_2(features_1_2)
        ret2 = self.scale_1_2_3(features_1_2_3)
        return ret1, ret2
    
    @staticmethod
    def task_loss(logits, targets):
        loss = []
        criterion = torch.nn.BCELoss()
        for i in range(len(logits)):
            loss.append(criterion(logits[i], targets))
        loss = torch.sum(torch.stack(loss))
        return loss

    def __echo_backbone(self, inputs, targets, optimizer_1_2, optimizer_1_2_3):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        scale_1_2, scale_1_2_3 = self.forward(inputs)
        optimizer_1_2.zero_grad()
        optimizer_1_2_3.zero_grad()
        # logit --> the vector of raw (non-normalized) predictions that a classification model generates
        loss_1_2 = self.task_loss(scale_1_2, targets)
        loss_1_2_3 = self.task_loss(scale_1_2, targets)
        loss_1_2.backward()
        loss_1_2_3.backward()
        optimizer_1_2.step()
        optimizer_1_2_3.step()
        #nb returning loss.item() is important to not saturate gpu memory!!!
        return loss_1_2.item(), loss_1_2_3.item()

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
    net = FusionCLS(num_classes=6)
    net.load_state_dict(torch.load(pretrained_model))
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
        fusion_loss_1_2, fusion_loss_1_2_3 = train(net, trainloader, fusion_1_2_opt, fusion_1_2_3_opt, epoch, 'backbone')
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
    cleanlog()  # dato che ora viene fatto append dei file, se non li elimini dopo ogni run accumula dati
    # path = 'logs'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # RACNN con backbone e apn pre addestrate
    run(pretrained_model='build/racnn_pretrained.pt')

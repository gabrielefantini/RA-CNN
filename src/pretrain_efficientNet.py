import os
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from plant_loader import get_plant_loader


def log(msg):
    open('build/core.log', 'a').write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]\t'+msg+'\n'), print(msg)

def eval(net, dataloader):
    log(' :: Testing on test set ...')
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            logits = net(inputs)
            correct_top1 += torch.eq(logits.topk(max((1, 1)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top3 += torch.eq(logits.topk(max((1, 3)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top5 += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()

        if step > 200:
            log(f'\tAccuracy@top1 ({step}/{len(dataloader)}) = {correct_top1/((step+1)*int(inputs.shape[0])):.5%}')
            log(f'\tAccuracy@top3 ({step}/{len(dataloader)}) = {correct_top3/((step+1)*int(inputs.shape[0])):.5%}')
            log(f'\tAccuracy@top5 ({step}/{len(dataloader)}) = {correct_top5/((step+1)*int(inputs.shape[0])):.5%}')
            return

def run():
    state_dict = torchvision.models.efficientnet_b0(pretrained=True).state_dict()
    state_dict.pop('classifier.1.weight')
    state_dict.pop('classifier.1.bias')
    net = torchvision.models.efficientnet_b0(num_classes=6).cuda()

    state_dict['classifier.1.weight'] = net.state_dict()['classifier.1.weight']
    state_dict['classifier.1.bias'] = net.state_dict()['classifier.1.bias']
    net.load_state_dict(state_dict)
    cudnn.benchmark = True

    #for param in net.parameters():
    #    print(type(param), param.size())

    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    data_set = get_plant_loader()

    trainloader = DataLoader(data_set["train"], batch_size=40, shuffle=True)
    validationloader = DataLoader(data_set["validation"], batch_size=40, shuffle=False)
    

    for epoch in range(100):  # loop over the dataset multiple times
            losses = 0
            
            for step, (inputs, labels) in enumerate(trainloader, 0):
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                optimizer.zero_grad()
                outputs = net(inputs)

                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                losses += loss
                if step % 20 == 0 and step != 0:
                    avg_loss = losses/20
                    log(f':: loss @step({step:2d}/{len(trainloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
                    losses = 0
            eval(net, validationloader)
            if epoch % 20 == 0 and epoch != 0:
                stamp = f'e{epoch}{int(time.time())}'
                torch.save(net, f'build/efficientNet_b0_ImageNet-{stamp}.pt')
                torch.save(optimizer.state_dict, f'build/optimizer-{stamp}.pt')


if __name__ == "__main__":
    path = 'build'
    if not os.path.exists(path):
        os.makedirs(path)
    run()

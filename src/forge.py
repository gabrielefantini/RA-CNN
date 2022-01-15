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


def test(net, dataloader, type):
    log(f' :: Testing on {type} set ...')
    accuracy = 0
    inputsNumber = 0

    correct_summary = {
        'clsf-0': {
            'top-1': 0,
            },
        'clsf-1': {
            'top-1': 0,
        },
        'clsf-2': {
            'top-1': 0,
            }
        }

    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        inputsNumber += inputs.size(0)

        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            for idx, logits in enumerate(outputs):
                logits = torch.sigmoid(logits)
                logits[logits >= 0.5 ] = 1
                logits[logits < 0.5 ] = 0
                correct_summary[f'clsf-{idx}']['top-1'] +=  torch.all(torch.eq(logits, labels),  dim=1).sum()  # top-1
                #correct_summary[f'clsf-{idx}']['top-5'] += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-5
            for clsf in correct_summary.keys():
                _summary = correct_summary[clsf]
                for topk in _summary.keys():
                    
                    if step == 100:
                        print(f'\tAccuracy {clsf}@{topk} {_summary[topk]/inputsNumber:.5%}')
                    
                    accuracy +=_summary[topk]/inputsNumber
    
    return accuracy/3

#impostazioni attuali --> 20 minuti ad epoch --> 16h per fare tutto l'addestramento
def run(pretrained_model):
    accuracy = 0
    log(f' :: Start training with {pretrained_model}')
    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    cudnn.benchmark = True

    #ognuna delle 3 cnn del modello parte con i valori della cnn pre addestrata e poi ognuna si specializza
    #con i propri parametri
    cls_params = list(net.b1.parameters()) + list(net.b2.parameters()) + list(net.b3.parameters()) + \
        list(net.classifier1.parameters()) + list(net.classifier2.parameters()) + list(net.classifier3.parameters()) #addestro sia le cnn che i classificatori
    apn_params = list(net.apn1.parameters()) + list(net.apn2.parameters())

    cls_opt = optim.SGD(cls_params, lr=0.001, momentum=0.9)
    #TODO da modificare in lr=1e-6
    apn_opt = optim.SGD(apn_params, lr=1e-6, momentum=0.9)

    data_set = get_plant_loader()
    trainloader = torch.utils.data.DataLoader(data_set["train"], batch_size=10, shuffle=True, num_workers=4)
    validationloader = torch.utils.data.DataLoader(data_set["validation"], batch_size=10, shuffle=False)
    sample = random_sample(validationloader)

    for epoch in range(50):
        net.train()
        cls_loss = train(net, trainloader, cls_opt, epoch, 'backbone')
        rank_loss = train(net, trainloader, apn_opt, epoch, 'apn')
        net.eval()
        temp_accuracy = test(net, validationloader, 'validation')
        train_accuracy = test(net, trainloader, 'train')

        print(f'avg Validation accuracy: {temp_accuracy}')
        print(f'avg train accuracy: {train_accuracy}')
        
        # visualize cropped inputs
        _, _, _, resized = net(sample.unsqueeze(0))
        x1, x2 = resized[0].data, resized[1].data
        save_img(x1, path=f'build/.cache/epoch_{epoch}@2x.jpg', annotation=f'cls_loss = {cls_loss:.7f}, rank_loss = {rank_loss:.7f}')
        save_img(x2, path=f'build/.cache/epoch_{epoch}@4x.jpg', annotation=f'cls_loss = {cls_loss:.7f}, rank_loss = {rank_loss:.7f}')


        if temp_accuracy > accuracy:
            accuracy = temp_accuracy
            stamp = f'e{epoch}{int(time.time())}'
            torch.save(net.state_dict(), f'build/racnn_efficientNetB0.pt')
            log(f' :: Saved model dict as:\tbuild/racnn_efficientNetB0.pt')
            torch.save(cls_opt.state_dict(), f'build/cls_optimizer.pt')
            torch.save(apn_opt.state_dict(), f'build/apn_optimizer.pt')


if __name__ == "__main__":
    clean()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #RACNN con backbone e apn pre addestrate
    run(pretrained_model='build/racnn_pretrained.pt')
    build_gif(pattern='@2x', gif_name='racnn_efficientNet_LRe')
    build_gif(pattern='@4x', gif_name='racnn_efficientNet_LRe')

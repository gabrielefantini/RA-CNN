import imageio
import os
import shutil
import sys
import numpy as np
from scipy.sparse import data
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
from fusion import FusionCLS


sys.path.append('.')  # noqa: E402
from model import RACNN
from plant_loader import get_plant_loader
from pretrain_apn import log, clean, save_img, build_gif

'''
def random_sample(dataloader):
    for batch_idx, (inputs, labels) in enumerate(dataloader, 1):
        return [inputs[5].cuda(), labels[5].cuda()]

def runOnSingleImage(pretrained_model):
    labels = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]

    net = RACNN(num_classes=6).cuda()
    net.eval()
    net.load_state_dict(torch.load(pretrained_model))
    data_set = get_plant_loader()
    validationloader = torch.utils.data.DataLoader(data_set["validation"], batch_size=32, shuffle=False)
    
    sample = random_sample(validationloader)
    preds, _, _, resized = net(sample[0].unsqueeze(0))
    
    #print(preds)
    #print(sample[1])

    for id, pred in enumerate(preds, 0):
        preds[id] = torch.sigmoid(preds[id])
        preds[id][preds[id] >= 0.5 ] = 1
        preds[id][preds[id] < 0.5 ] = 0

    x1, x2 = resized[0].data, resized[1].data
    save_img(sample[0].unsqueeze(0), path=f'build/.cache/original.jpg', annotation=f'test')
    save_img(x1, path=f'build/.cache/test@2x.jpg', annotation=f'test')
    save_img(x2, path=f'build/.cache/test@4x.jpg', annotation=f'test')
    
    print("Image label: ")
    for id, lab in enumerate(sample[1], 0):
        if(lab.item()==1):
            print(labels[id]) 
    print("================================")
    for id, pred in enumerate(preds, 0):
        print(f'Pred number{id}:')
        for id, x in enumerate(pred[0], 0):
            if(x.item()==1):
                print(labels[id])
        print("================================")        
'''

def run(pretrained_model):
    accuracy = 0
    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    net.eval()
    data_set = get_plant_loader()
    validationloader = torch.utils.data.DataLoader(data_set["validation"], batch_size=32, shuffle=False)
    categories = ["complex", "frog_eye_leaf_spot",
              "healthy", "powdery_mildew", "rust", "scab"]
    
    total_true= Variable().cuda()
    preds = {
    'clsf-0': Variable().cuda(),
    'clsf-1': Variable().cuda(),
    'clsf-2': Variable().cuda()
    }

    for step, (inputs, labels) in enumerate(validationloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        total_true = torch.cat((total_true, labels), dim=0)
        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            #gli output di ogni livello
            for idx, logits in enumerate(outputs):
                logits = torch.sigmoid(logits)
                logits[logits >= 0.5 ] = 1
                logits[logits < 0.5 ] = 0
                preds[f'clsf-{idx}'] = torch.cat((preds[f'clsf-{idx}'], logits), dim=0)
                
    for i in range(3):
        print(f'Scale number {i}')
        print(metrics.classification_report(total_true.cpu(), preds[f'clsf-{i}'].cpu(), target_names=categories))

def runFusion(pretrained_model):
    accuracy = 0
    net = FusionCLS(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    net.eval()
    data_set = get_plant_loader()
    validationloader = torch.utils.data.DataLoader(data_set["validation"], batch_size=32, shuffle=False)
    categories = ["complex", "frog_eye_leaf_spot",
              "healthy", "powdery_mildew", "rust", "scab"]
    
    total_true= Variable().cuda()
    preds = {
    'scale-0': Variable().cuda(),
    'scale-1': Variable().cuda()
    }

    for step, (inputs, labels) in enumerate(validationloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        total_true = torch.cat((total_true, labels), dim=0)
        with torch.no_grad():
            outputs= net(inputs)
            #gli output di ogni livello
            for idx, logits in enumerate(outputs):
                logits[logits >= 0.5 ] = 1
                logits[logits < 0.5 ] = 0
                preds[f'scale-{idx}'] = torch.cat((preds[f'scale-{idx}'], logits), dim=0)


    print('Scale 1_2')
    print(metrics.classification_report(total_true.cpu(), preds['scale-0'].cpu(), target_names=categories))
    print('Scale 1_2_3')
    print(metrics.classification_report(total_true.cpu(), preds['scale-1'].cpu(), target_names=categories))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #runOnSingleImage('build/racnn_efficientNetB0.pt')
    #run('build//racnn_efficientNetB0.pt')
    runFusion('build//fusion_efficientNetB0.pt')

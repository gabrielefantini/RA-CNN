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

def run(pretrained_model):
    accuracy = 0
    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run('build/racnn_efficientNetB0.pt')
    #build_gif(pattern='@2x', gif_name='racnn_efficientNet')
    #build_gif(pattern='@4x', gif_name='racnn_efficientNet')

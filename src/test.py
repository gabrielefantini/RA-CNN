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
from pretrain_apn import log, clean, save_img, build_gif

def random_sample(dataloader):
    for batch_idx, (inputs, labels) in enumerate(dataloader, 1):
        return [inputs[0].cuda(), labels[0].cuda()]

def run(pretrained_model):
    accuracy = 0
    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))

    data_set = get_plant_loader()
    validationloader = torch.utils.data.DataLoader(data_set["validation"], batch_size=32, shuffle=False)
    sample = random_sample(validationloader)
    preds, _, _, resized = net(sample[0].unsqueeze(0))
    x1, x2 = resized[0].data, resized[1].data
    save_img(sample[0].unsqueeze(0), path=f'build/.cache/original.jpg', annotation=f'test')
    save_img(x1, path=f'build/.cache/test@2x.jpg', annotation=f'test')
    save_img(x2, path=f'build/.cache/test@4x.jpg', annotation=f'test')
    print(sample[1], preds)



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run('build/racnn_efficientNetB0.pt')
    #build_gif(pattern='@2x', gif_name='racnn_efficientNet')
    #build_gif(pattern='@4x', gif_name='racnn_efficientNet')

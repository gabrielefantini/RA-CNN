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
import matplotlib.pyplot as plt
import random


sys.path.append('.')  # noqa: E402
from model import RACNN
from plant_loader import get_plant_loader
from pretrain_apn import log, clean, save_img, build_gif


def random_sample(dataloader):
    for batch_idx, (inputs, labels, paths) in enumerate(dataloader, 1):
        return [inputs[19].cuda(), labels[19].cuda()]


def runOnSingleImage(pretrained_model):
    labels = ["complex", "frog_eye_leaf_spot",
              "healthy", "powdery_mildew", "rust", "scab"]

    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    data_set = get_plant_loader()
    validationloader = torch.utils.data.DataLoader(
        data_set["validation"], batch_size=32, shuffle=False)

    sample = random_sample(validationloader)

    original = sample[0].unsqueeze(0)
    save_img(original, path=f'build/.cache/original.jpg', annotation=f'test')

    preds, _, _, resized = net(original)

    # print(preds)
    # print(sample[1])

    for id, pred in enumerate(preds, 0):
        preds[id] = torch.sigmoid(preds[id])
        preds[id][preds[id] >= 0.5] = 1
        preds[id][preds[id] < 0.5] = 0

    x1, x2 = resized[0].data, resized[1].data
    save_img(x1, path=f'build/.cache/test@2x.jpg', annotation=f'test')
    save_img(x2, path=f'build/.cache/test@4x.jpg', annotation=f'test')

    print("Image label: ")
    for id, lab in enumerate(sample[1], 0):
        if(lab.item() == 1):
            print(labels[id])
    print("================================")
    for id, pred in enumerate(preds, 0):
        print(f'Pred number{id}:')
        for id, x in enumerate(pred[0], 0):
            if(x.item() == 1):
                print(labels[id])
        print("================================")


def getLabelString(x, id):
    labels = ["complex", "frog_eye_leaf_spot",
              "healthy", "powdery_mildew", "rust", "scab"]
    if(x == 1):
        return labels[id]


def getRandom(dataloader, batchId, label, nsamples):
    samples = []

    for id, (inputs, labels, paths) in enumerate(dataloader, 1):
        if id == batchId:
            # batch found
            for sampleId in range(0, len(inputs)):
                # iterating samples
                valid = True
                labels_ = []
                for i, x in enumerate(labels[sampleId]):
                    l = getLabelString(x.item(), i)
                    if l is not None:
                        labels_.append(l)
                for l in labels_:
                    if l != label:
                        valid = False
                if valid is True:
                    # print(paths[sampleId], ": valid")
                    samples.append(
                        [inputs[sampleId].cuda(), labels[sampleId].cuda(), paths[sampleId]])
                # else:
                #     print(paths[sampleId], ": not valid")
                if len(samples) == nsamples:
                    return samples

    return samples


def getSamplesByLabel(dataloader, nsamples, label):
    samples = []

    nbatches = len(dataloader)
    ids = []
    for i in range(1, nbatches+1):
        ids.append(i)
    random.shuffle(ids)

    while len(samples) != nsamples:
        id = ids.pop(0)
        requiredSamples = nsamples - len(samples)
        # print(f'searching {requiredSamples} {label} samples in batch {id}')
        samples += getRandom(dataloader, id, label, requiredSamples)
    return samples


def debug():
    # per provare roba
    return


def runTest(pretrained_model):
    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    dataset = get_plant_loader()
    dataloader = torch.utils.data.DataLoader(
        dataset["validation"], batch_size=32, shuffle=False)

    data_labels = ["complex", "frog_eye_leaf_spot",
                   "healthy", "powdery_mildew", "rust", "scab"]

    for data_label in data_labels:
        print(data_label, ":")
        for sample_id, sample in enumerate(getSamplesByLabel(dataloader, 5, data_label), 0):
            print("================================")
            print(f'sample: {data_label}-{sample_id}\n')

            labels = []

            for id, x in enumerate(sample[1]):
                l = getLabelString(x.item(), id)
                if l is not None:
                    labels.append(l)

            labelstring = "{ "
            for label in labels:
                labelstring += label+" "
            labelstring += "}"

            print(sample[2]+" : "+labelstring)

            original = sample[0].unsqueeze(0)
            save_img(
                original, path=f'build/.cache/og-{data_label}-{sample_id}.jpg', annotation=f'test')

            net.eval()
            preds, _, _, resized = net(original)

            for id, pred in enumerate(preds, 0):
                preds[id] = torch.sigmoid(preds[id])
                preds[id][preds[id] >= 0.5] = 1
                preds[id][preds[id] < 0.5] = 0

            x1, x2 = resized[0].data, resized[1].data
            save_img(
                x1, path=f'build/.cache/test-{data_label}-{sample_id}@2x.jpg', annotation=f'test')
            save_img(
                x2, path=f'build/.cache/test-{data_label}-{sample_id}@4x.jpg', annotation=f'test')

            for id, pred in enumerate(preds, 0):
                print(f'Pred number{id}:')
                # print(pred)
                labels = []
                for id, x in enumerate(pred[0], 0):
                    l = getLabelString(x.item(), id)
                    if l is not None:
                        labels.append(l)
                labelstring = "{ "
                for label in labels:
                    labelstring += label+" "
                labelstring += "}"
                print(labelstring)

            print("================================\n")
        print("\n\n")


def run(pretrained_model):
    accuracy = 0
    net = RACNN(num_classes=6).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    net.eval()
    data_set = get_plant_loader()
    validationloader = torch.utils.data.DataLoader(
        data_set["validation"], batch_size=32, shuffle=False)

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

    inputsNumber = 0
    for step, (inputs, labels) in enumerate(validationloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        inputsNumber += inputs.size(0)

        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            # gli output di ogni livello
            for idx, logits in enumerate(outputs):
                logits = torch.sigmoid(logits)
                logits[logits >= 0.5] = 1
                logits[logits < 0.5] = 0
                correct_summary[f'clsf-{idx}']['top-1'] += torch.all(
                    torch.eq(logits, labels),  dim=1).sum()  # top-1

    for clsf in correct_summary.keys():
        _summary = correct_summary[clsf]
        for topk in _summary.keys():
            print(
                f'\tAccuracy {clsf}@{topk} {_summary[topk]/inputsNumber:.5%}')
            accuracy += _summary[topk]/inputsNumber

    print(accuracy/3)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    clean()
    runTest('build/racnn_efficientNetB0.pt')
    # runOnSingleImage('build/racnn_efficientNetB0.pt')
    # run('build/racnn_efficientNetB0.pt')

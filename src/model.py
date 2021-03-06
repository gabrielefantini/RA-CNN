from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

#Utils function
def unravel_index(
        indices: torch.LongTensor,
        shape: tuple[int, ...],
    ) -> torch.LongTensor:
        r"""Converts flat indices into unraveled coordinates in a target shape.

        This is a `torch` implementation of `numpy.unravel_index`.

        Args:
            indices: A tensor of indices, (*, N).
            shape: The targeted shape, (D,).

        Returns:
            unravel coordinates, (*, N, D).
        """

        shape = torch.tensor(shape)
        indices = indices % shape.prod()  # prevent out-of-bounds indices

        coord = torch.zeros(indices.size() + shape.size(), dtype=int)

        for i, dim in enumerate(reversed(shape)):
            coord[..., i] = indices % dim
            indices = indices // dim

        return coord.flip(-1)

class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, apn_out, inputs):
        h = lambda x: 1 / (1 + torch.exp(-10 * x))
        unit = torch.stack([torch.arange(0, 224)] * 224)
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(inputs, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()
        
        ret = []
        for i in range(inputs.size(0)):
            tx = 112 + int(apn_out[i][0] * 38 )
            ty = 112 + int(apn_out[i][1] * 38 )
            #tl = 28 + int(apn_out[i][2] * 28 + 0.5)
            #tl da 38 a 56 (Prova 1) ovvero da 1/3 a 1/2
            #tl da 38 a 56 (Prova 2) ovvero da 1/3 a 1/2
            #tl da 45 a 74 (Prova 3) ovvero da 2/5 a 2/3
            tl = 45 + int(((apn_out[i][2] + 1) /2 ) * 29)
            
            mk = (h(x-tx+tl) - h(x-tx-tl)) * (h(y-ty+tl) - h(y-ty-tl))
            xatt = inputs[i] * mk
            
            xatt_cropped = xatt[:, tx-tl : tx+tl, ty-tl : ty+tl]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.upsample(before_upsample, size=(224,224), mode='bilinear')
            ret.append(xamp.data.squeeze())
        
        ret_tensor = torch.stack(ret)
        self.save_for_backward(inputs, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(ctx, grad_output):
        inputs, ret_tensor = ctx.saved_variables[0], ctx.saved_variables[1]
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        
#         show_image(inputs.cpu().data[0])
#         show_image(ret_tensor.cpu().data[0])
#         plt.imshow(norm[0].cpu().numpy(), cmap='gray')
        
        x = torch.stack([torch.arange(0, 224)] * 224).t()
        y = x.t()
        mx = (x >= 150).float() - (x < 74).float()
        my = (y >= 150).float() - (y < 74).float()
        ml = (((x<74)+(x>=150)+(y<74)+(y>=150)) > 0).float()*2 - 1
        
        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))
        
        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()
        
        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        
        return ret, None



class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, apn_out, inputs):
        return AttentionCropFunction.apply(apn_out, inputs)


class RACNN(nn.Module):
    def __init__(self, num_classes):
        super(RACNN, self).__init__()

        self.b1 = torchvision.models.efficientnet_b0(num_classes=num_classes)
        self.b2 = torchvision.models.efficientnet_b0(num_classes=num_classes)
        self.b3 = torchvision.models.efficientnet_b0(num_classes=num_classes)
                
        state_dict = torch.load('build/efficientNet_b0_ImageNet.pt').state_dict()
        #state_dict.pop('classifier.1.weight')
        #state_dict.pop('classifier.1.bias')
        #eff = torchvision.models.efficientnet_b0(num_classes=6).cuda()
        #state_dict['classifier.1.weight'] = eff.state_dict()['classifier.1.weight']
        #state_dict['classifier.1.bias'] = eff.state_dict()['classifier.1.bias']

        self.b1.load_state_dict(state_dict)
        self.b2.load_state_dict(state_dict)
        self.b3.load_state_dict(state_dict)

        self.bf1 = self.b1.features[:-1]
        self.bf2 = self.b2.features[:-1]
        self.bf3 = self.b3.features[:-1]

        self.classifier1 = nn.Sequential(
            self.b1.features[8],
            self.b1.avgpool,
            nn.Flatten(),
            self.b1.classifier
        )
        self.classifier2 = nn.Sequential(
            self.b2.features[8],
            self.b2.avgpool,
            nn.Flatten(),
            self.b2.classifier
        )
        self.classifier3 = nn.Sequential(
            self.b3.features[8],
            self.b3.avgpool,
            nn.Flatten(),
            self.b3.classifier
        )
        '''
        for param in self.bf1.parameters():
            param.requires_grad = False
        for param in self.bf2.parameters():
            param.requires_grad = False
        for param in self.bf3.parameters():
            param.requires_grad = False
        '''
        self.crop_resize = AttentionCropLayer()

        self.apn1 = nn.Sequential(
            nn.Linear(320 * 7 * 7, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 3),
            nn.Tanh()
        )

        self.apn2 = nn.Sequential(
            nn.Linear(320 * 7 * 7, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 3),
            nn.Tanh()
        )
        
        self.echo = None

    def forward(self, x):
        #==========================================================
        #                       Scale_1
        #==========================================================
        feature_s1 = self.bf1(x)  # torch.Size([1, 320, 7, 7])
        pred1 = self.classifier1(feature_s1)

        apn_out1 = self.apn1(feature_s1.view(-1, 320 * 7 * 7))
        resized_s1 = self.crop_resize(apn_out1, x)

        #==========================================================
        #                       Scale_2
        #==========================================================
        feature_s2 = self.bf2(resized_s1)  # torch.Size([1, 320, 7, 7])
        pred2 = self.classifier2(feature_s2)

        apn_out2 = self.apn2(feature_s2.view(-1, 320 * 7 * 7))
        resized_s2 = self.crop_resize(apn_out2, resized_s1)
        
        #==========================================================
        #                       Scale_3
        #==========================================================
        feature_s3 = self.bf3(resized_s2)
        pred3 = self.classifier3(feature_s3)
        
        return [pred1, pred2, pred3], [feature_s1, feature_s2], [apn_out1, apn_out2], [resized_s1, resized_s2]
        
    def __get_weak_loc(self, features):
        ret = []   # search regions with the highest response value
        for i in range(len(features)):
            #features[i].size() == [nbatch, 320, 7, 7]
            #Upscala il 7x7 a 224x224 e poi fa la media tra i canali
            response_map_batch = F.interpolate(features[i], size=[224, 224], mode="bilinear").amax(1)
            #response_map_batch.size() == [nbatch, 224, 224]
            ret_batch = []
            for response_map in response_map_batch:
                #response_map.size() = [224,224]
                tx, ty = unravel_index(response_map.argmax(), response_map.shape)
                #tx, ty possono andare da 56 a 168, tl fissato a 56
                tx.clamp(min=56, max=168)
                ty.clamp(min=56, max=168)
                tx = (tx - 112)/56
                ty = (ty - 112)/56
                ret_batch.append([tx, ty, 1])
            ret.append(torch.Tensor(ret_batch))
        return ret

    def __echo_pretrain_apn(self, inputs, optimizer):
        inputs = Variable(inputs).cuda()
        _, features, attens, _ = self.forward(inputs)
        weak_loc = self.__get_weak_loc(features)
        optimizer.zero_grad()
        weak_loss1 = F.smooth_l1_loss(attens[0], weak_loc[0].cuda())
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1].cuda())
        loss = weak_loss1 + weak_loss2
        loss.backward()
        optimizer.step()
        return loss.item()

    @staticmethod
    def multitask_loss(logits, targets):
        loss = []
        criterion1 = torch.nn.BCEWithLogitsLoss()
        criterion2 = torch.nn.BCEWithLogitsLoss()
        criterion3 = torch.nn.BCEWithLogitsLoss()
        return criterion1(logits[0], targets), criterion2(logits[1], targets), criterion3(logits[2], targets)
    
    @staticmethod
    def rank_loss(logits, targets, margin=0.05):
        preds = [torch.sigmoid(x) for x in logits] # preds length equal to 3
        criterion1 = torch.nn.MarginRankingLoss(margin=0.05, reduction='sum')
        criterion2 = torch.nn.MarginRankingLoss(margin=0.05, reduction='sum')
        y = torch.tensor([-1]).cuda()
        return criterion1(preds[0], preds[1], y), criterion2(preds[1], preds[2], y)
    
    def __echo_backbone(self, inputs, targets, optimizers):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optim1, optim2, optim3 = optimizers
        loss1, loss2, loss3 = self.multitask_loss(logits, targets)
        
        optim3.zero_grad()
        loss3.backward(retain_graph=True)
        optim3.step()

        optim2.zero_grad()
        loss2.backward(retain_graph=True)
        optim2.step()

        optim1.zero_grad()
        loss1.backward()
        optim1.step()

        return loss1.item()+loss2.item()+loss3.item()

    def __echo_apn(self, inputs, targets, optimizers):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optim1, optim2 = optimizers
        loss1, loss2 = self.rank_loss(logits, targets)
        
        optim2.zero_grad()
        loss2.backward(retain_graph=True)
        optim2.step()

        optim1.zero_grad()
        loss1.backward()
        optim1.step()

        return loss1.item()+loss2.item()

    def mode(self, mode_type):
        assert mode_type in ['pretrain_apn', 'apn', 'backbone']
        if mode_type == 'pretrain_apn':
            self.echo = self.__echo_pretrain_apn
            self.eval()
        if mode_type == 'backbone':
            self.echo = self.__echo_backbone
            self.train()
        if mode_type == 'apn':
            self.echo = self.__echo_apn
            self.eval()

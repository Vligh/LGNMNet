#!/usr/bin/env python
# coding=utf-8
"""
#!/usr/bin/python3
-*- coding: utf-8 -*-
@Time    : 2021/9/2 下午4:46
@Author  : xsbank
@Email   : yangsai1991@163.com

"""

import os
import cv2
import time
import copy
import shutil
import argparse
import torch
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler

import warnings
import json

warnings.filterwarnings("ignore")
from utils.utils import load_label
from utils.find_result import ParameterOptimization, ParameterOptimizationAll
from model.AverageMeter import AverageMeter
from data.train_dataset2 import ImageDatasetImg, ImageDatasetImgVal
from model.head.head_def import HeadFactory
from model.backbone.backbone_def import BackboneFactory

from loguru import logger
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, backbone_factory, head_factory, conf):
        """Init face model by backbone factorcy and head factory.

        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.args = conf
        self.backbone = backbone_factory.get_backbone()
        if self.args.resume:
            model_dict = self.backbone.state_dict()
            pretrain_model = './weights/MobileFaceNet.pt'
            logger.info('load pre-train model:%s' % pretrain_model)
            pretrained_dict = torch.load(pretrain_model)['state_dict']
            new_pretrained_dict = {}
            for k in model_dict:
                new_pretrained_dict[k] = pretrained_dict['backbone.' + k]
            model_dict.update(new_pretrained_dict)
            self.backbone.load_state_dict(model_dict)
        if self.args.bool_head:
            self.head_factory = head_factory
            self.head = head_factory.get_head()
            self.Linear = nn.Sequential(
                nn.Dropout(0.2),  # 0.2
                nn.Linear(backbone_factory.backbone_param['feat_dim'], head_factory.head_param['feat_dim']),
                # nn.ReLU6(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Linear(head_factory.head_param['feat_dim'], head_factory.head_param['num_class']), )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(backbone_factory.backbone_param['feat_dim'], backbone_factory.backbone_param['num_class']), )

        # x = torch.randn(1, 3, 112, 112)  # 输入
        # from thop import profile
        # flops, params = profile(self.backbone, inputs=(x,))
        # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
        # print("params=", str(params / 1e6) + '{}'.format("M"))

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        if self.args.bool_head:
            feat = F.normalize(feat)
            feat = self.Linear(feat)
            pred, loss = self.head.forward(feat, label)
            return pred, loss
        else:
            pred = self.classifier(feat)
            return pred

    def predict(self, data):
        feat = self.backbone.forward(data)
        if self.args.bool_head:
            feat = F.normalize(feat)
            feat = self.Linear(feat)
            pred = self.head.predict(feat)
        else:
            pred = self.classifier(feat)
        return pred


def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def inference(net, image, device, input_size=[112, 112]):
    image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
    image_ = torch.from_numpy(image.astype(np.float32))
    image_ = image_[np.newaxis,]
    image_ = image_.to(device)
    outputs = net.predict(image_)
    prediction = F.softmax(outputs).detach().cpu().numpy()
    label = np.argmax(prediction)
    value = prediction[0][label]
    return label, value

def eval(model, args, uid, express='Micro'):
    model.eval()
    info_path = '%s/%s/%s/label/%s_val.json' % (args.data_root, args.dataset, express, uid)
    with open(info_path) as json_file:
        video_info = json.load(json_file)
    json_file.close()
    one_pre = []
    for video_id in video_info:
        annotations = video_info[video_id]['annotations']
        npy_path = '%s/%s/%s/npy/%s.npy' % (args.data_root, args.dataset, express, video_id)
        testset = ImageDatasetImgVal(npy_path)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
        predictions = []
        for batch_idx, (images) in enumerate(testloader):
            time.sleep(0.15)
            images = images.to(device)
            outputs = model.module.predict(images)
            prediction = F.softmax(outputs).detach().cpu().numpy().tolist()
            predictions.extend(prediction)
        preds = []
        for ii in predictions:
            label = np.argmax(ii)
            value = ii[label]
            preds.append({'label': label, 'value': value})
        one_pre.append({'id': video_id, 'result': preds, 'label': annotations})
    return one_pre


def train_one_epoch(data_loader, model, optimizer, criterion, loss_meter, conf, cur_epoch):
    """Tain one epoch by traditional training.
    """
    model.train()  # Set model to training mode
    for batch_idx, (images, labels) in enumerate(data_loader):
        time.sleep(0.15)
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.bool_head:
            outputs, loss_g = model.forward(images, labels)
            loss_g = torch.mean(loss_g)
            loss = criterion(outputs, labels) + loss_g
        else:
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        loss_meter.reset()
    return model


def train(conf, total_dataloader, uid, all_pres, Parameter, boolOptimization=False):
    """Total training procedure.
    """
    conf.device = torch.device(device)
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device)
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file)
    model = FaceModel(backbone_factory, head_factory, conf)
    model = torch.nn.DataParallel(model).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]

    # optimizer = optim.Adam(parameters, lr=conf.lr)
    optimizer = optim.SGD(parameters, lr=conf.lr, momentum=0.9, weight_decay=5e-4)
    # lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=0.1)
    lr_schedule = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1)
    loss_meter = AverageMeter()

    best_model_wts = copy.deepcopy(model.state_dict())
    iterNum, best_fscore, best_one_pre = 0, 0, []
    for epoch in range(conf.epoches):
        model = train_one_epoch(total_dataloader, model, optimizer, criterion, loss_meter, conf, epoch)
        one_pre = eval(model, args, uid, express=args.mode)
        all_pre = all_pres.copy()
        all_pre.extend(one_pre)
        if boolOptimization:
            ParameterResult = ParameterOptimizationAll(args, all_pre)
            Parameter = ParameterResult
        else:
            ParameterResult = ParameterOptimization(args, all_pre, Parameter)
        # oneResult = ParameterOptimization(args, one_pre, ParameterResult)
        fscore = ParameterResult['f_score']
        print('fscore:', ParameterResult, 'epoch:', epoch)
        if best_fscore < fscore:
            iterNum = 0
            best_fscore = fscore.copy()
            best_one_pre = one_pre.copy()
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = '%s/%s' % (conf.weight, uid)
            if not os.path.exists(model_path): os.makedirs(model_path)
            torch.save(model.state_dict(), '%s/model.pth' % (model_path))  # param
            save_json = '%s/Parameter.json' % (model_path)
            json_file = open(save_json, mode='w')
            json.dump(ParameterResult, json_file, indent=4)
            logger.info(
                '----------------%s Epoch: %d Parameter: %s ----------------' % (uid, epoch, ParameterResult))
        else:
            # model.load_state_dict(best_model_wts)
            iterNum += 1
            if iterNum > 3: break
        lr_schedule.step()
    all_pres.extend(best_one_pre)
    return all_pres, Parameter

def sort_sub(args, all_subjects):
    sort_num = []
    for uid in all_subjects:
        info_path = '%s/%s/%s/label/%s_val.json' % (args.data_root, args.dataset, args.mode, uid)
        with open(info_path) as json_file:
            video_info = json.load(json_file)
        json_file.close()
        num = 0
        for i in video_info:
            num += len(video_info[i]['annotations'])
        sort_num.append(num)
    index = np.argsort(-np.array(sort_num))
    sort_subjects = np.array(all_subjects)[index].tolist()
    return sort_subjects

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--K', type=int, default=6,
                        help='parameter K, samm-micro:37, samm-macro:174, cas-micro:6, cas-macro:18', metavar='N')
    parser.add_argument('--mode', type=str, default="Micro",
                        help='Micro/Macro', metavar='N')
    parser.add_argument('--dataset', type=str, default="CAS",
                        help="SAMM/CAS", metavar='N')
    parser.add_argument('--data_root', type=str,
                        default="/media/sai/data1/datasets/face/emotion/npy",
                        help='the path of saving preprocess data', metavar='N')
    parser.add_argument('--backbone_type', type=str, default='MobileNet',
                        help='MobileFaceNet, MobileNet, ShuffleNet.')
    parser.add_argument('--backbone_conf_file', type=str, default='./model/backbone_conf.yaml',
                        help='the path of backbone_conf.yaml.')
    parser.add_argument("--head_type", type=str, default='MagFace',
                        help="MagFace, MV-Softmax ...")
    parser.add_argument("--head_conf_file", type=str, default='./model/head_conf.yaml',
                        help="the path of head_conf.yaml..")
    parser.add_argument("--local_rank", type=int, default=0, help="multi=[0, 1], single=0")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The initial learning rate.MoorSide39')
    parser.add_argument('--epoches', type=int, default=3,
                        help='The training epoches.')
    parser.add_argument('--milestones', type=int, default=[3, 5, 7],
                        help='10, 13, 16')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size over all gpus.')
    parser.add_argument("--bool_head", type=bool, default=False)
    parser.add_argument('--resume', '-r', default=False,
                        help='Whether to resume from a checkpoint.')
    args = parser.parse_args()

    logger.add('log/%s_%s.log' % (args.dataset, args.mode), encoding='utf-8')

    args.weight = './weights/MEGC/%s_%s/mobilenet' % (args.dataset, args.mode)
    if not os.path.exists(args.weight): os.makedirs(args.weight)
    args.path_xlsx = './input/%s.xlsx' % (args.dataset)

    _, _, label_micro, path_micro, all_subjects = load_label(args.path_xlsx, args.dataset, args.mode)
    all_subjects = sort_sub(args, all_subjects)
    logger.info(args)

    all_pres = []
    # train
    count, boolOptimization, Parameter, pre_Parameter = 0, True, None, None

    for uid in all_subjects:
        info_path = '%s/%s/%s/label/%s_val.json' % (args.data_root, args.dataset, args.mode, uid)
        with open(info_path) as json_file:
            video_info = json.load(json_file)
        json_file.close()
        # data_loader
        trainset = ImageDatasetImg(args.data_root, args.dataset, uid, augment=False, express=args.mode, mode='train',
                                   K=args.K)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                  drop_last=True)
        all_pres, Parameter = train(args, trainloader, uid, all_pres, Parameter, boolOptimization=boolOptimization)
        if pre_Parameter == None:
            pre_Parameter = Parameter
        elif Parameter['P'] == pre_Parameter['P'] and Parameter['WIN'] == pre_Parameter['WIN'] and Parameter['ORDER'] == \
                pre_Parameter['ORDER'] and Parameter['TINY'] == pre_Parameter['TINY']:
            count += 1
        else:
            count = 0
            pre_Parameter = Parameter
        if count > 4: boolOptimization = False

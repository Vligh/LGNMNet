# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import warnings

warnings.filterwarnings("ignore")
from utils.calculate import *
from utils.find_result import *
from data.train_dataset import ImageDatasetImgVal
from model.head.head_def import HeadFactory
from model.backbone.backbone_def import BackboneFactory


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
                nn.Linear(512, 2), )

    def predict(self, data):
        feat = self.backbone.forward(data)
        if self.args.bool_head:
            feat = F.normalize(feat)
            feat = self.Linear(feat)
            pred = self.head.predict(feat)
        else:
            pred = self.classifier(feat)
        return pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--K', type=int, default=37,
                        help='parameter K, samm-micro:37, samm-macro:174, cas-micro:6, cas-macro:18', metavar='N')
    parser.add_argument('--mode', type=str, default="Micro",
                        help='CAS:micro-expression/macro-expression SAMM:Micro/Macro', metavar='N')
    parser.add_argument('--dataset', type=str, default="SAMM",
                        help="SAMM/CAS", metavar='N')
    parser.add_argument('--data_root', type=str, default="/media/sai/data1/datasets/face/emotion/npy",
                        help='the path of saving preprocess data', metavar='N')
    parser.add_argument('--backbone_type', type=str, default='MobileNet',
                        help='MobileFaceNet, MobileNet.')
    parser.add_argument('--backbone_conf_file', type=str, default='./model/backbone_conf.yaml',
                        help='the path of backbone_conf.yaml.')
    parser.add_argument("--bool_head", type=bool, default=True)
    parser.add_argument("--head_type", type=str, default='MagFace',
                        help="mv-softmax, arcface, npc-face ...")
    parser.add_argument("--head_conf_file", type=str, default='./model/head_conf.yaml',
                        help="the path of head_conf.yaml..")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.weight = './weights/MEGC/%s_%s/src' % (args.dataset, args.mode)
    if not os.path.exists(args.weight): os.makedirs(args.weight)
    args.path_xlsx = './input/%s.xlsx' % (args.dataset)

    _, _, label_micro, path_micro, all_subjects = load_label(args.path_xlsx, args.dataset, args.mode)

    # test
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    head_factory = HeadFactory(args.head_type, args.head_conf_file)
    model = FaceModel(backbone_factory, head_factory, args)
    # model.cuda(device)
    model = torch.nn.DataParallel(model).cuda()

    all_pres = []
    for uid in all_subjects:
        print('uid:', uid)
        info_path = '%s/%s/%s/label/%s_val.json' % (args.data_root, args.dataset, args.mode, uid)
        with open(info_path) as json_file:
            video_info = json.load(json_file)
        json_file.close()
        model_path = '%s/%s/model.pth' % (args.weight, uid)
        if not os.path.exists(model_path):
            for video_id in video_info:
                annotations = video_info[video_id]['annotations']
                all_pres.append({'id': video_id, 'result': [], 'label': annotations})
            continue
        model.load_state_dict(torch.load(model_path))
        model.eval()
        for video_id in video_info:
            annotations = video_info[video_id]['annotations']
            npy_path = '%s/%s/%s/npy/%s.npy' % (args.data_root, args.dataset, args.mode, video_id)
            testset = ImageDatasetImgVal(npy_path)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
            predictions = []
            for batch_idx, (images) in enumerate(testloader):
                images = images.to(device)
                outputs = model.module.predict(images)
                prediction = F.softmax(outputs).detach().cpu().numpy().tolist()
                predictions.extend(prediction)
            preds = []
            for ii in predictions:
                label = np.argmax(ii)
                value = ii[label]
                preds.append({'label': label, 'value': value})
            all_pres.append({'id': video_id, 'result': preds, 'label': annotations})
    ParameterResult = ParameterOptimizationAll(args, all_pres)
    print(ParameterResult)
    # with open('%s/model_parameter.json' % (args.weight), 'w') as file_obj:
    #     json.dump(json.dumps(ParameterResult), file_obj)
    # file_obj.close()

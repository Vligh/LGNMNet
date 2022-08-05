import os, sys
import numpy as np
import cv2

import os.path as osp
# from PIL import Image
import argparse
from multiprocessing import Manager, Pool
from utils.utils import extract_preprocess, cal_k, load_label


def multi_process(args, labelList):
    for i in labelList:
        label, lmks = i['id'], i['bbox_lmk']
        print(label)
        file_path = '%s/%s' % (args.root_path, label)
        # imgList = os.listdir(file_path)
        # imgList.sort()
        num = len(lmks)
        flow_file = '%s/%s/%s/flow' % (args.path_save, args.dataset, args.mode)
        if not os.path.exists(flow_file):
            os.makedirs(flow_file)
        # flow_num = len(os.listdir(os.path.join(flow_file, label.split('/')[-1])))
        # file_num = len(os.listdir(file_path))
        # if flow_num+args.K>=file_num:
        #     continue

        os.system(
            'denseflow {} -a={} -b={} -st={} -s={} -o={} -v --if'.format(file_path, args.algorithm, args.bound, 'png',
                                                                         args.K, flow_file))  # tvl1,farn
        sys.stdout.flush()

        npy_file = '%s/%s/%s/npy' % (args.path_save, args.dataset, args.mode)
        if not os.path.exists(npy_file):
            os.makedirs(npy_file)
        save_path = '%s/%s_%d.npy' % (npy_file, label.split('/')[-1][:7], num)
        imgs = []
        for j in range(num):
            if lmks[j] is None:
                continue
            img_path = '%s/%s/flow_%05d.png' % (flow_file, label.split('/')[-1], j)
            img = cv2.imread(img_path)
            if img is None:
                continue
            final_image = extract_preprocess(img, lmks[j]['lmk'], img_size=args.lmk_size)
            final_image = final_image.astype(np.uint8)
            img = cv2.resize(final_image, (args.img_size, args.img_size))
            imgs.append(img[:, :, ::-1])
        imgs = np.stack(imgs)  # bgr
        # imgs = np.stack(imgs, 0)  # flow
        if imgs.shape[0] > args.max_frame_num:
            imgs = imgs[:args.max_frame_num]
        np.save(save_path, imgs)


def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset', default='CAS', type=str, help="SAMM/CAS/SMIC")
    parser.add_argument('--root_path', default='/media/sai/data1/datasets/face/emotion/CASME2/align_112',
                        type=str)
    parser.add_argument('--path_save', default='/media/sai/data1/datasets/face/emotion/npy', type=str)
    parser.add_argument('--num_workers', default=0, type=int, help='num of workers to act multi-process')
    parser.add_argument('--img_size', default=112, type=int, help='gap frames')
    parser.add_argument('--lmk_size', default=112, type=int, help='gap frames')
    parser.add_argument('--max_frame_num', default=10e5, type=int, help='gap frames')
    parser.add_argument('--bound', default=15, type=int, help='set the maximum of optical flow')
    parser.add_argument('--mode', default='Micro', type=str)
    parser.add_argument('--algorithm', default='tvl1', type=str, help='tvl1,farn')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    path_xlsx = '../input/%s.xlsx' % (args.dataset)
    _, _, label_micro, path_micro, all_subjects = load_label(path_xlsx, args.dataset, args.mode)
    args.K = cal_k(label_micro)

    labelList = np.load(os.path.join(args.root_path, 'label.npy'), allow_pickle=True).tolist()
    multi_process(args, labelList)

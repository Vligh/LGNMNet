import os,sys
import numpy as np
import cv2
from PIL import Image
import argparse
from multiprocessing import Manager, Pool
from utils.utils import extract_preprocess, cal_k, load_label

def flow2bgr(flows, to_gray=False):
    # crate HSV & make Value a constant
    h, w, channel = flows.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flows[..., 0], flows[..., 1])
    # Use Hue and Saturation to encode the Optical Flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def multi_process(args, labelList):
    # dtvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    step = args.K
    for i in labelList:
        label, lmks = i['id'], i['bbox_lmk']
        print(label)
        file_path = '%s/%s' % (args.root_path, label)
        imgList = os.listdir(file_path)
        imgList.sort()
        num = len(lmks)
        if not os.path.exists(args.path_save):
            os.makedirs(args.path_save)
        # else:continue

        save_path = '%s/%s_%d.npy' % (args.path_save, label.split('/')[-1][:7], num)
        imgs = []
        for j in range(num-step):
            if lmks[j] is None or lmks[j+step] is None:
                continue
            img_path1 = os.path.join(file_path, imgList[j])
            img_path2 = os.path.join(file_path, imgList[j+step])
            img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
            ##default choose the tvl1 algorithmcalc
            # flows = dtvl1.calc(img1, img2, None)
            flows = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.702, 5, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            # flows = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # bgr
            flows = flow2bgr(flows)
            final_image = extract_preprocess(flows, lmks[j]['lmk'], img_size=args.lmk_size)
            # img = cv2.resize(final_image, (96, 96))
            # flow_x, flow_y
            flow_x = ToImg(final_image[..., 0], args.bound)
            flow_y = ToImg(final_image[..., 1], args.bound)
            # save_x = '%s/flow_x_%04d.jpg' % (save_file, j + 1)
            # save_y = '%s/flow_y_%04d.jpg' % (save_file, j + 1)
            # cv2.imwrite(save_x, flow_x)
            # cv2.imwrite(save_y, flow_y)
            flow_x = cv2.resize(flow_x, (args.img_size, args.img_size))
            flow_y = cv2.resize(flow_y, (args.img_size, args.img_size))
            img = np.stack([flow_x, flow_y], -1)
            img = (img).astype(np.uint8)
            imgs.append(img[:, :, ::-1])
        # imgs = np.stack(imgs) # bgr
        imgs = np.stack(imgs, 0)  # flow
        if imgs.shape[0] > args.max_frame_num:
            imgs = imgs[:args.max_frame_num]
        np.save(save_path, imgs)

def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset',default='CAS',type=str,help="SAMM/CAS/SMIC")
    parser.add_argument('--root_path',default='/media/sai/data1/datasets/face/emotion/CAS/align_224',type=str)
    parser.add_argument('--path_save',default='/media/sai/data1/datasets/face/emotion/npy',type=str)
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--img_size',default=96,type=int,help='gap frames')
    parser.add_argument('--lmk_size',default=224,type=int,help='gap frames')
    parser.add_argument('--max_frame_num',default=10e4,type=int,help='gap frames')
    parser.add_argument('--bound',default=15,type=int,help='set the maximum of optical flow')
    parser.add_argument('--mode',default='Micro',type=str)
    parser.add_argument('--type',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args

if __name__ =='__main__':

    args=parse_args()
    path_xlsx = '../../datasets/%s.xlsx' % (args.dataset)
    label_micro, path_micro, all_subjects = load_label(path_xlsx, args.dataset, args.mode)
    args.K = cal_k(label_micro)

    args.path_save = '%s/%s/%s/npy' % (args.path_save, args.dataset, args.mode)
    labelList = np.load(os.path.join(args.root_path, 'label.npy'), allow_pickle=True).tolist()
    if args.mode=='run':
        num_thread = args.num_workers
        num_lenght = int(len(labelList) / num_thread)

        manager = Manager()
        lock = manager.Lock()
        pool = Pool(processes=num_thread)

        for ii in range(0, len(labelList), num_lenght):
            pool.apply_async(multi_process, (args, labelList[ii:ii+num_lenght],))
        pool.close()
        pool.join()
    else:
        multi_process(args, labelList)

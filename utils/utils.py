import json
import pandas as pd
import numpy as np
import cv2
import os

def cal_k(final_samples):
    final_samples = final_samples
    total_duration, num = 0, 0
    for i in final_samples:
        for j in i:
            num += 1
            total_duration += j[1] - j[0]

    N = total_duration / num
    k = int((N + 1) / 2)
    print('k (Half of average length of expression) =', k)
    return k


def read_xlsx_samm(path, expression_type):
    sheet = pd.read_excel(path)
    all_labels = []
    all_paths = []
    for i in sheet.values[:]:
        if i[3] != 0:
            all_labels.append([i[3], i[5]])
        else:
            all_labels.append([i[4], i[5]])
        all_paths.append(i[1].split("_")[0] + "_" + i[1].split("_")[1])
    path_set = list(set(all_paths))
    path_set.sort()
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)
    all_path_list = []
    all_label_intervals = []
    for one_path in path_set:
        all_path_list.append(one_path)
        all_label_intervals.append(all_labels[np.where(all_paths == one_path)])

    labels, paths = [], []
    for i in sheet.values[:]:
        if expression_type in i[7]:
            if i[3] != 0:
                labels.append([i[3], i[5]])
            else:
                labels.append([i[4], i[5]])
            paths.append(i[1].split("_")[0] + "_" + i[1].split("_")[1])
    path_set = list(set(paths))
    path_set.sort()
    paths = np.array(paths)
    labels = np.array(labels)
    path_list = []
    label_intervals = []
    for one_path in path_set:
        path_list.append(one_path)
        label_intervals.append(labels[np.where(paths == one_path)])
    return all_label_intervals, all_path_list, label_intervals, path_list


def read_xlsx_cas(path, expression_type):
    sheet = pd.read_excel(path)
    name_rule_2 = pd.read_excel(path,
                                sheet_name="naming rule2")
    name_rule_1 = pd.read_excel(path,
                                sheet_name="naming rule1")

    name_1 = {}
    for i in name_rule_1.values:
        name_1[i[2]] = i[0]
    name_2 = {}
    name_2["disgust1"] = 101
    for i in name_rule_2.values:
        name_2[i[1]] = i[0]
    allList = []
    for i in range(len(sheet.values)):
        allList.append(sheet.values[i])
    all_path_list = []
    all_label_intervals, all_label_interval = [], []
    pre_sub = None
    for i in range(len(allList)):
        a = name_2[allList[i][1].split("_")[0]]
        b = name_1[allList[i][0]]
        cur_sub = 's%s/%s_0%s*' % (b, b, a)
        if (pre_sub != cur_sub) and pre_sub != None:
            all_label_intervals.append(all_label_interval)
            all_label_interval = []
            all_path_list.append(pre_sub)
        if allList[i][3] > allList[i][4]:
            all_label_interval.append([allList[i][2], allList[i][3]])
        else:
            all_label_interval.append([allList[i][2], allList[i][4]])
        pre_sub = cur_sub
        if i == len(allList) - 1:
            all_label_intervals.append(all_label_interval)
            all_path_list.append(pre_sub)

    valueList = []
    for i in range(len(sheet.values)):
        express = sheet.values[i][7]
        if expression_type == express:
            value = sheet.values[i]
            valueList.append(value)

    path_list = []
    label_intervals, label_interval = [], []
    pre_sub = None
    for i in range(len(valueList)):
        a = name_2[valueList[i][1].split("_")[0]]
        b = name_1[valueList[i][0]]
        cur_sub = 's%s/%s_0%s*' % (b, b, a)
        if (pre_sub != cur_sub) and pre_sub != None:
            label_intervals.append(label_interval)
            label_interval = []
            path_list.append(pre_sub)
        if valueList[i][3] > valueList[i][4]:
            label_interval.append([valueList[i][2], valueList[i][3]])
        else:
            label_interval.append([valueList[i][2], valueList[i][4]])
        pre_sub = cur_sub
        if i == len(valueList) - 1:
            label_intervals.append(label_interval)
            path_list.append(pre_sub)
    return all_label_intervals, all_path_list, label_intervals, path_list

def read_xlsx_smic(path, expression_type):
    sheet = pd.read_excel(path)
    all_label = []
    all_path = []
    for i in sheet.values[:]:
        label = []
        idx, subject, video, video_name, NumME, StartFrame, EndFrame = i[:7]
        for j in range(NumME):
            onset, offset = i[3 * j + 8] - StartFrame, i[3 * j + 9] - StartFrame
            label.append([onset, offset])
        all_path.append(video_name + '*')
        all_label.append(label)

    return all_label, all_path, all_label, all_path

def load_label(path_xlsx, dataset, mode):
    # load data
    if dataset == "SAMM":
        all_label, all_path, label, path = read_xlsx_samm(path_xlsx, mode)
        all_subjects = list(set([i.split("_")[0] for i in path]))
    elif dataset == "CAS":
        all_label, all_path, label, path = read_xlsx_cas(path_xlsx, mode)
        all_subjects = list(set([i.split("/")[-1].split("_")[0] for i in path]))
    else:
        all_label, all_path, label, path = read_xlsx_smic(path_xlsx, mode)
        all_subjects = list(set([i.split("_")[0] for i in path]))
    all_subjects.sort()
    return all_label, all_path, label, path, all_subjects


def cal_IOU(interval_1, interval_2):
    intersection = [max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])]
    union_set    = [min(interval_1[0], interval_2[0]), max(interval_1[1], interval_2[1])]
    if intersection[0]<=intersection[1]:
        len_inter = intersection[1]-intersection[0]+1
        len_union = union_set[1]-union_set[0]+1
        return len_inter/len_union
    else:
        return 0


def extract_preprocess(flow_img, lmk, img_size=112):
    height, weight, channel = flow_img.shape
    # Use Hue and Saturation to encode the Optical Flow
    shape = lmk[:, :2]

    if img_size == 112:
        beta1, beta2 = 4, 6
    elif img_size == 224:
        beta1, beta2 = 8, 12
    else:
        beta1, beta2 = 16, 24
    # Left Eye
    x11 = max(shape[66, 0] - beta1, 0)
    y11 = shape[66, 1]
    x12 = shape[67, 0]
    y12 = max(shape[67, 1] - beta1, 0)
    x13 = shape[68, 0]
    y13 = max(shape[68, 1] - beta1, 0)
    x14 = shape[69, 0]
    y14 = max(shape[69, 1] - beta1, 0)
    x15 = min(shape[70, 0] + beta1, weight)
    y15 = shape[70, 1]
    x16 = shape[71, 0]
    y16 = min(shape[71, 1] + beta1, height)
    x17 = shape[72, 0]
    y17 = min(shape[72, 1] + beta1, height)
    x18 = shape[73, 0]
    y18 = min(shape[73, 1] + beta1, height)

    # Right Eye
    x21 = max(shape[75, 0] - beta1, 0)
    y21 = shape[75, 1]
    x22 = shape[76, 0]
    y22 = max(shape[76, 1] - beta1, 0)
    x23 = shape[77, 0]
    y23 = max(shape[77, 1] - beta1, 0)
    x24 = shape[78, 0]
    y24 = max(shape[78, 1] - beta1, 0)
    x25 = min(shape[79, 0] + beta1, weight)
    y25 = shape[79, 1]
    x26 = shape[80, 0]
    y26 = min(shape[80, 1] + beta1, height)
    x27 = shape[81, 0]
    y27 = min(shape[81, 1] + beta1, height)
    x28 = shape[82, 0]
    y28 = min(shape[82, 1] + beta1, height)

    # ROI 1 (Eyebrow)
    x31 = max(min(shape[33:50, 0]) - beta1, 0)
    y31 = max(min(shape[33:50, 1]) - beta1, 0)
    x32 = min(max(shape[33:50, 0]) + beta1, weight)
    y32 = min(max(shape[66:82, 1]) + beta1, height)

    # ROI 2 #Mouth
    x51 = max(min(shape[84:95, 0]) - beta2, 0)
    y51 = max(min(shape[84:95, 1]) - beta1, 0)
    x52 = min(max(shape[84:95, 0]) + beta2, weight)
    y52 = min(max(shape[84:95, 1]) + beta1, height)

    # Nose landmark
    x61 = shape[52, 0]
    y61 = shape[52, 1]

    # Remove global head movement by minus nose region
    flow_img[:, :, 0] = abs(flow_img[:, :, 0] - flow_img[y61 - beta1:y61 + beta2, x61 - beta1:x61 + beta2, 0].mean())
    flow_img[:, :, 1] = abs(flow_img[:, :, 1] - flow_img[y61 - beta1:y61 + beta2, x61 - beta1:x61 + beta2, 1].mean())
    if channel > 2:
        flow_img[:, :, 2] = flow_img[:, :, 2] - flow_img[y61 - beta1:y61 + beta2, x61 - beta1:x61 + beta2, 2].mean()
    # Eye masking
    left_eye = [(x11, y11), (x12, y12), (x13, y13), (x14, y14), (x15, y15), (x16, y16), (x17, y17), (x18, y18)]
    right_eye = [(x21, y21), (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26), (x27, y27), (x28, y28)]
    cv2.fillPoly(flow_img, [np.array(left_eye)], 0)
    cv2.fillPoly(flow_img, [np.array(right_eye)], 0)
    # ROI Selection -> Image resampling into 42x22x3
    top, botton = 56, 112
    final_image = np.zeros((112, 112, channel))
    final_image[:top, :, :] = cv2.resize(flow_img[y31: y32, x31:x32, :], (botton, top))
    final_image[top:botton, :, :] = cv2.resize(flow_img[y51:y52, x51:x52, :], (botton, top))
    return final_image



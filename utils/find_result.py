# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, find_peaks_cwt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.calculate import cal_f1_score, spotting_evaluation_V2


def softnms_v2(segments, sigma=0.5, top_k=10, score_threshold=0.1):
    import torch
    segments = torch.tensor(segments)
    segments = segments.cpu()
    tstart = segments[:, 0]
    tend = segments[:, 1]
    tscore = segments[:, 2]
    done_mask = tscore < -1  # set all to False
    undone_mask = tscore >= score_threshold
    while undone_mask.sum() > 1 and done_mask.sum() < top_k:
        idx = tscore[undone_mask].argmax()
        idx = undone_mask.nonzero()[idx].item()

        undone_mask[idx] = False
        done_mask[idx] = True

        top_start = tstart[idx]
        top_end = tend[idx]
        _tstart = tstart[undone_mask]
        _tend = tend[undone_mask]
        tt1 = _tstart.clamp(min=top_start)
        tt2 = _tend.clamp(max=top_end)
        intersection = torch.clamp(tt2 - tt1, min=0)
        duration = _tend - _tstart
        tmp_width = torch.clamp(top_end - top_start, min=1e-5)
        iou = intersection / (tmp_width + duration - intersection)
        scales = torch.exp(-iou ** 2 / sigma)
        tscore[undone_mask] *= scales
        undone_mask[tscore < score_threshold] = False
    segments = torch.stack([tstart[done_mask], tend[done_mask], tscore[done_mask]], -1)

    segments = segments.detach().cpu().numpy().tolist()
    new_segments = []
    for segment in segments:
        new_segments.append([int(segment[0]), int(segment[1]), segment[2]])
    return new_segments


def recursive_merge(inter, start_index=0):
    for i in range(start_index, len(inter) - 1):
        if inter[i][1] >= inter[i + 1][0]:
            new_start = int(0.5 * (inter[i][0] + inter[i + 1][0]))
            new_end = int(0.5 * (inter[i][1] + inter[i + 1][1]))
            new_score = max(inter[i][2], inter[i + 1][2])
            # new_start = int(
            #     inter[i][0] * inter[i][2] / (inter[i][2] + inter[i + 1][2]) + inter[i + 1][0] * inter[i + 1][2] / (
            #                 inter[i][2] + inter[i + 1][2]))
            # new_end = int(
            #     inter[i][1] * inter[i][2] / (inter[i][2] + inter[i + 1][2]) + inter[i + 1][1] * inter[i + 1][2] / (
            #                 inter[i][2] + inter[i + 1][2]))
            # new_score = max(inter[i][2], inter[i + 1][2])
            inter[i] = [new_start, new_end, new_score]
            del inter[i + 1]
            return recursive_merge(inter.copy(), start_index=i)
    return inter


def post_process(args, result, P=0.55, win=11, order=8, tiny=0.7, rate=4, bool_peak=False):
    if len(result) < 1:
        return []
    Y = []
    for j in range(len(result)):
        Y.append(result[j]['label'] * result[j]['value'])
    Y = np.array(Y)
    Y_POST = savgol_filter(Y, win, order)
    threshold = np.mean(Y_POST) + P * (max(Y_POST) - np.mean(Y_POST))  # Moilanen threshold technique
    peaks, _ = list(find_peaks(Y_POST, height=threshold, distance=args.K))
    peaks_pair = []
    if len(peaks) < 1:
        peaks_pair = []
    elif len(peaks) == 1:
        onset, offset = int(max(0, peaks[0] - int(tiny * args.K))), int(min(peaks[0] + int(tiny * args.K), len(Y_POST)))
        peaks_pair.append([onset, offset, np.mean(Y_POST[onset:offset])])
    else:
        if bool_peak:
            for ii in peaks:
                onset, offset = int(max(0, ii - int(tiny * args.K))), int(
                    min(ii + int(tiny * args.K), len(Y_POST)))
                peaks_pair.append([onset, offset, np.mean(Y_POST[onset:offset])])
        else:
            for i in range(len(peaks) - 1):
                onset, offset = int(max(0, peaks[i] - int(tiny * args.K))), int(
                    min(peaks[i + 1] + int(tiny * args.K), len(Y_POST)))
                if offset - onset > rate * args.K: continue
                peaks_pair.append([onset, offset, np.mean(Y_POST[onset:offset])])
        if len(peaks_pair) > 1:
            peaks_pair = softnms_v2(peaks_pair)
            peaks_pair = recursive_merge(sorted(peaks_pair))
    return peaks_pair


def ParameterOptimization(args, results, parameter):
    one_subject_TP, one_subject_FP, one_subject_FN = 0, 0, 0
    for i in results:
        result, label = i['result'], i['label']
        pred_inter = post_process(args, result, P=parameter['P'], win=parameter['WIN'], order=parameter['ORDER'],
                                  tiny=parameter['TINY'])
        if len(pred_inter) < 1:
            TP, FP, FN = 0, 0, len(label)
        else:
            TP, FP, FN = spotting_evaluation_V2(pred_inter, label)
        one_subject_TP += TP
        one_subject_FP += FP
        one_subject_FN += FN
    if one_subject_TP == 0:
        f1_score = 0
    else:
        recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP,
                                                   one_subject_FN)

    ParameterResult = {'f_score': f1_score, 'P': parameter['P'], 'WIN': parameter['WIN'], 'ORDER': parameter['ORDER'],
                       'TINY': parameter['TINY'],
                       'TP': int(one_subject_TP),
                       'FP': int(one_subject_FP), 'FN': int(one_subject_FN)}
    return ParameterResult


def ParameterOptimizationAll(args, results):
    best_tiny, best_p, best_win, best_order, best_nms = 0, 0, 0, 0, 0
    max_f_score, best_TP, best_FP, best_FN = 0, 0, 0, 0
    for tiny in np.linspace(0.1, 0.6, num=6):
        for p in np.linspace(0.6, 1, num=5):
            for win in [9, 11, 13, 15]:
                for order in np.linspace(0, 4, num=5):
                    f1_score = 0
                    one_subject_TP, one_subject_FP, one_subject_FN = 0, 0, 0
                    for i in results:
                        result, label = i['result'], i['label']
                        pred_inter = post_process(args, result, P=p, win=win, order=int(order), tiny=tiny)
                        if len(pred_inter) < 1:
                            TP, FP, FN = 0, 0, len(label)
                        else:
                            TP, FP, FN = spotting_evaluation_V2(pred_inter, label)
                        one_subject_TP += TP
                        one_subject_FP += FP
                        one_subject_FN += FN
                    if one_subject_TP > 0:
                        recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP,
                                                                   one_subject_FN)
                    if max_f_score <= f1_score or (
                            max_f_score == 0 and best_FP + best_FN > one_subject_FP + one_subject_FN) or (
                            max_f_score == 0 and best_FP == 0 and best_FN == 0):
                        max_f_score, best_p, best_win, best_order, best_tiny = f1_score, p, win, order, tiny
                        best_TP, best_FP, best_FN = one_subject_TP, one_subject_FP, one_subject_FN

    ParameterResult = {'f_score': max_f_score, 'P': float(best_p), 'WIN': int(best_win), 'ORDER': int(best_order),
                       'TINY': float(best_tiny),
                       'TP': int(best_TP),
                       'FP': int(best_FP), 'FN': int(best_FN)}
    return ParameterResult


def ParameterOptimizationHAHA(args, results):
    TINY = {}
    for tiny in np.linspace(0.1, 1., num=10
                            ):
        P = {}
        for p in np.linspace(0.1, 1, num=10):
            one_subject_TP, one_subject_FP, one_subject_FN = 0, 0, 0
            for i in results:
                result, label = i['result'], i['label']
                pred_inter = post_process(args, result, P=p, win=13, order=4, tiny=tiny)
                if len(pred_inter) < 1:
                    TP, FP, FN = 0, 0, len(label)
                else:
                    TP, FP, FN = spotting_evaluation_V2(pred_inter, label)
                one_subject_TP += TP
                one_subject_FP += FP
                one_subject_FN += FN
            recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP,
                                                           one_subject_FN)
            P[p] = f1_score
        TINY[tiny] = P
    return TINY


def plot_result(args, result, parameter, rate=4, bool_peak=False):
    P = parameter['P']
    win = parameter['WIN']
    order = parameter['ORDER']
    tiny = parameter['TINY']
    if len(result) < 1:
        return []
    Y = []
    for j in range(len(result)):
        Y.append(result[j]['label'] * result[j]['value'])
    Y = np.array(Y)
    Y_POST = savgol_filter(Y, win, order)
    threshold = np.mean(Y_POST) + P * (max(Y_POST) - np.mean(Y_POST))  # Moilanen threshold technique
    peaks, _ = list(find_peaks(Y_POST, height=threshold, distance=args.K))
    peaks_pair = []
    if len(peaks) < 1:
        peaks_pair = []
    elif len(peaks) == 1:
        onset, offset = int(max(0, peaks[0] - int(tiny * args.K))), int(min(peaks[0] + int(tiny * args.K), len(Y_POST)))
        peaks_pair.append([onset, offset, np.mean(Y_POST[onset:offset])])
    else:
        if bool_peak:
            for ii in peaks:
                onset, offset = int(max(0, ii - int(tiny * args.K))), int(
                    min(ii + int(tiny * args.K), len(Y_POST)))
                peaks_pair.append([onset, offset, np.mean(Y_POST[onset:offset])])
        else:
            for i in range(len(peaks) - 1):
                onset, offset = int(max(0, peaks[i] - int(tiny * args.K))), int(
                    min(peaks[i + 1] + int(tiny * args.K), len(Y_POST)))
                if offset - onset > rate * args.K: continue
                peaks_pair.append([onset, offset, np.mean(Y_POST[onset:offset])])
        if len(peaks_pair) > 1:
            peaks_pair = softnms_v2(peaks_pair)
            peaks_pair = recursive_merge(sorted(peaks_pair))
    return Y, Y_POST, threshold, peaks, peaks_pair

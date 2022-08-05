import numpy as np
from scipy import signal

from utils.utils import *

def cal_TP(left_count_1, label):
    result = []
    for inter_2 in label:
        temp = 0
        for inter_1 in left_count_1:
            if cal_IOU(inter_1, inter_2)>=0.5:
                temp += 1
        result.append(temp)
    return result

def spotting_evaluation(pred, express_inter, K, P):
    pred = np.array(pred)
    threshold = np.mean(pred)+ P*(np.max(pred)-np.mean(pred))
    num_peak = signal.find_peaks(pred, height=threshold, distance=K*2)
    pred_inter = []
    
    for peak in num_peak[0]:
        pred_inter.append([peak-K, peak+K])

    result = cal_TP(pred_inter, express_inter)
    result = np.array(result)
    TP = len(np.where(result!=0)[0])
    n = len(pred_inter)-(sum(result)-TP)
    m = len(express_inter)
    FP = n-TP
    FN = m-TP

    return TP, FP, FN, pred_inter

def spotting_evaluation_V2(pred_inter, express_inter):
    result = cal_TP(pred_inter, express_inter)
    result = np.array(result)
    TP = len(np.where(result!=0)[0])
    n = len(pred_inter)-(sum(result)-TP)
    m = len(express_inter)
    FP = n-TP
    FN = m-TP
    return TP, FP, FN

def cal_f1_score(TP, FP, FN):
    if TP == 0:
        recall, precision, f1_score = 0, 0, 0
    else:
        recall = TP/(TP+FP)
        precision = TP/(TP+FN)
        f1_score = 2*recall*precision/(recall+precision)
    return recall, precision, f1_score


def get_auc(labels, preds):
    # 这段代码基本上是沿着公式计算的：
    # 1. 先求正样本的rank和
    # 2. 再减去（m*(m+1)/2）
    # 3. 最后除以组合个数

    # 但是要特别注意，需要对预测值pred相等的情况进行了一些处理。
    # 对于这些预测值相等的样本，它们对应的rank是要取平均的

    # 先将data按照pred进行排序
    sorted_data = sorted(list(zip(labels, preds)), key=lambda item: item[1])
    pos = 0.0  # 正样本个数
    neg = 0.0  # 负样本个数
    auc = 0.0
    # 注意这里的一个边界值，在初始时我们将last_pre记为第一个数，那么遍历到第一个数时只会count++
    # 而不会立刻向结果中累加（因为此时count==0，根本没有东西可以累加）
    last_pre = sorted_data[0][1]
    count = 0.0
    pre_sum = 0.0  # 当前位置之前的预测值相等的rank之和，rank是从1开始的，所以在下面的代码中就是i+1
    pos_count = 0.0  # 记录预测值相等的样本中标签是正的样本的个数

    # 为了处理这些预测值相等的样本，我们这里采用了一种lazy计算的策略：
    # 当预测值相等时仅仅累加count，直到下次遇到一个不相等的值时，再将他们一起计入结果
    for i, (label, pred) in enumerate(sorted_data):
        # 注意：rank就是i+1
        if label > 0:
            pos += 1
        else:
            neg += 1
        if last_pre != pred:  # 当前的预测概率值与前一个值不相同
            # lazy累加策略被触发，求平均并计入结果，各个累积状态置为初始态
            auc += pos_count * pre_sum / count  # 注意这里只有正样本的部分才会被累积进结果
            count = 1
            pre_sum = i + 1  # 累积rank被清空，更新为当前数rank
            last_pre = pred
            if label > 0:
                pos_count = 1  # 如果当前样本是正样本 ，则置为1
            else:
                pos_count = 0  # 反之置为0
        # 如果预测值是与前一个数相同的，进入累积状态
        else:
            pre_sum += i + 1  # rank被逐渐累积
            count += 1  # 计数器也被累计
            if label > 0:  # 这里要另外记录正样本数，因为负样本在计算平均
                pos_count += 1  # rank的时候会被计入，但不会被计入rank和的结果

    # 注意这里退出循环后我们要额外累加一下。
    # 这是因为我们上面lazy的累加策略，导致最后一组数据没有累加
    auc += pos_count * pre_sum / count
    auc -= pos * (pos + 1) / 2  # 减去正样本在正样本之前的情况即公式里的(m+1)m/2
    auc = auc / (pos * neg)  # 除以总的组合数即公式里的m*n
    return auc
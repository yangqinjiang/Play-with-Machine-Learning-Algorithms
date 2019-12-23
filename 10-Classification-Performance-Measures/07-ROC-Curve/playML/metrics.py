import numpy as np
from math import sqrt

def accuracy_score(y_true,y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0],\
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) /len(y_true)


def mean_squared_error(y_true,y_predict):
    """计算y_true与y_predict之间的MSE"""
    assert len(y_true) == len(y_predict),\
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_predict - y_true) ** 2 ) / len(y_true)

def root_mean_squared_error(y_true,y_predict):
    """计算y_true与y_predict之间的RMSE"""

    return  sqrt(mean_squared_error(y_true,y_predict))

def mean_absolute_error(y_true,y_predict):
    """计算y_true和y_predict之间的MAE"""
    assert len(y_true) == len(y_predict),\
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true,y_predict):
    """计算y_true和y_predict之间的R^2"""
    assert len(y_true) == len(y_predict),\
        "the size of y_true must be equal to the size of y_predict"

    return  1 - mean_squared_error(y_true, y_predict) / np.var(y_true)


def TN(y_true,y_predict):
    assert len(y_true) == len(y_predict) # 数据长度
    # 实际数据与预测数据 都为 false, 即两者相同, 预测正确(T)
    return np.sum( (y_true == 0) & (y_predict == 0) )


def FP(y_true,y_predict):
    assert len(y_true) == len(y_predict) # 数据长度
    # 实际数据为0  但预测数据为1 (P), 预测错误(F)  
    return np.sum((y_true == 0)&(y_predict == 1))


def FN(y_true,y_predict):
    assert len(y_true) == len(y_predict) # 数据长度
    # 实际数据为 1  但预测数据为0 (N), 预测错误(F) 
    return np.sum((y_true == 1) & (y_predict == 0))


def TP(y_true,y_predict):
    assert len(y_true) == len(y_predict) # 数据长度
    # 实际数据为 1  预测数据为1 (P), 即两者相同,  预测正确(T) 
    return np.sum((y_true == 1) & (y_predict == 1))           

def confusion_matrix(y_true,y_predict):
    """创建混淆矩阵的函数"""
    return np.array([
        [TN(y_true,y_predict),FP(y_true,y_predict)],
        [FN(y_true,y_predict),TP(y_true,y_predict)]
    ])


def precision_score(y_true,y_predict):
    """精确率"""
    tp = TP(y_true,y_predict)
    fp = FP(y_true,y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0

def recall_score(y_true,y_predict):
    """召回率"""
    tp = TP(y_true,y_predict)
    fn = FN(y_true,y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def f1_score(y_true,y_predict):
    precision = precision_score(y_true,y_predict)
    recall = recall_score(y_true,y_predict)       
    try:
        return 2. * precision * recall / (precision + recall)
    except:
        return 0.0 

def TPR(y_true,y_predict):
    tp = TP(y_true,y_predict)
    fn = FN(y_true,y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0 

def FPR(y_true,y_predict):
    fp = FP(y_true,y_predict)
    tn = TN(y_true,y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0 
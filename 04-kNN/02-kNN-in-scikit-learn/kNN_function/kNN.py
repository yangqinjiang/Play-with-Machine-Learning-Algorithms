import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k,X_train,y_train,x):
    # k近邻的个数
    # 检查参数
    # k值必须在1~shape[0]之间
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    # 训练集的行数必须一致
    assert X_train.shape[0] == y_train.shape[0],\
    "the size of X_train must equal to the size of y_train"
    # 训练集与测试集的特征个数(列数)必须一致
    assert X_train.shape[1] == x.shape[0],\
    "the feature number of x must equal to X_train"

    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    #从小到大排序, 得到索引值
    nearest = np.argsort(distances)

    topK_y = [y_train[neighbor] for neighbor in nearest[:k]]
    votes = Counter(topK_y) #相同元素的计算
    predict_y = votes.most_common(1)[0][0] #获取一个最大值
    return predict_y
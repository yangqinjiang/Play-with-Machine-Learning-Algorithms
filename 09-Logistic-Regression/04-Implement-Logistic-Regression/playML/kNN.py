import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNClassifier:
    
    def __init__(self,k):
        """初始化kNN分类器"""
        assert k >= 1 , "k must be valid"
        self.k = k
        # 私有变量
        self._X_train = None
        self._y_train = None

    def fit(self, X_train,y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        # 训练集的行数必须一致
        assert X_train.shape[0] == y_train.shape[0],\
        "the size of X_train must equal to the size of y_train"
        # k数值不能大于训练集行数
        assert self.k <= X_train.shape[0],\
        "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self  #参考sklearn的设计
    
    def predict(self,X_predict):
        """给定等待预测数据X_predict,返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None,\
            "must fit before predict!"
        # 训练集与测试集的特征个数(列数)必须一致
        assert X_predict.shape[1] == self._X_train.shape[1],\
        "the feature number of X_predict must equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        """给定单个等待预测数据x,返回x的预测结果值"""
        # x是单行矩阵
        assert x.shape[0] == self._X_train.shape[1],\
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        #从小到大排序, 得到索引值
        nearest = np.argsort(distances)

        topK_y = [self._y_train[neighbor] for neighbor in nearest[:self.k]]
        votes = Counter(topK_y) #相同元素的计算
        predict_y = votes.most_common(1)[0][0] #获取一个最大值
        return predict_y
        
    def score(self,X_test,y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k
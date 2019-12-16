import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression 模型"""
        self.coef_ = None #系数
        self.intercept_ = None #截距
        self._theta = None #西塔


    def fit_normal(self,X_train,y_train):
        """根据训练数据集X_train,y_train训练Linear Regression模型"""
        # 行数必须一致
        assert X_train.shape[0] == y_train.shape[0] ,\
        "the size of X_train must be equal to the size of y_train"

        # 左边第一列,补充  len(X_train) 个 1
        X_b = np.hstack([np.ones((len(X_train), 1)) , X_train])

        # 根据公式, 向量化计算的
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        # 从_theta 矩阵中,拿到 intercept_ 和 coef_
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self,X_predict):
        """给定等待cbij数据集X_predict,返回表示X_predict的结果向量"""

        assert self.intercept_ is not None and self.coef_ is not None,\
        "must fit before predict"

        assert X_predict.shape[1] == len(self.coef_) ,\
        "the featurn number of X_predict must be equal to X_train"

        # 预测值
        X_b = np.hstack([np.ones((len(X_predict),1)),X_predict])
        return  X_b.dot(self._theta)

    def score(self,X_test,y_test):
        """根据预测数据集X_test,y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return  r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"
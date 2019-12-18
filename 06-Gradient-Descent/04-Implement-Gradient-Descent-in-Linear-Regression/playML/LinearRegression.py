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

    def fit_gd(self,X_train,y_train,eta=0.01,n_iters=1e4):
        """根据训练数据集X_train,y_train,使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0] ,\
        "the size of X_train must be equal to the size of y_train"

        
        def J(theta,X_b,y):
            """线性回归方程"""
            try:
                return np.sum( (y - X_b.dot(theta)) ** 2 ) / len(X_b)
            except:
                return float('inf')
            
        def dJ(theta,X_b,y):
            """对线性回归方程求导"""
            res = np.empty(len(theta))  # 创建一个临时的数组
            res[0] = np.sum(X_b.dot(theta) -y ) # 计算第一列的
            for i in range(1,len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b,y,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
            """适用于线性回归方程的梯度下降法"""
            theta = initial_theta
            cur_iter = 0
            
            # 循环次数
            while cur_iter < n_iters:
                gradient = dJ(theta,X_b,y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs( J(theta,X_b,y) - J(last_theta,X_b,y) ) < epsilon):
                    break
                cur_iter += 1
            return theta


        # 组装矩阵, 第一列是 1, 第二列之后是x
        X_b = np.hstack([ np.ones((len(X_train),1)),X_train.reshape(-1,1) ])
        # 初始化 theta 为 0
        initial_theta = np.zeros(X_b.shape[1])
        # eta 学习率
        self._theta = gradient_descent(X_b,y_train,initial_theta,eta)
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
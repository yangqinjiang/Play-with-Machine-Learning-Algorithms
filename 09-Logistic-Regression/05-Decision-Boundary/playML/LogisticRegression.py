import numpy as np
from .metrics import accuracy_score

class LogisticRegression:
    """逻辑回归"""
    def __init__(self):
        """初始化LogisticRegression 模型"""
        self.coef_ = None #系数
        self.intercept_ = None #截距
        self._theta = None #西塔

    def _sigmoid(self,t):
        """sigmoid函数: https://baike.baidu.com/item/Sigmoid%E5%87%BD%E6%95%B0/7981407"""
        return 1. / (1. + np.exp(-t))

    def fit(self,X_train,y_train,eta=0.01,n_iters=1e4):
        """[批量梯度下降法]根据训练数据集X_train,y_train,使用梯度下降法训练LogisticRegression模型"""
        assert X_train.shape[0] == y_train.shape[0] ,\
        "the size of X_train must be equal to the size of y_train"

        
        def J(theta,X_b,y):
            """逻辑回归的损失函数"""
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return np.sum( y*np.log(y_hat) + (1-y)*np.log(1-y_hat) ) / len(y)
            except:
                return float('inf')
            
        def dJ(theta,X_b,y):
            """对逻辑回归的损失函数求导"""
            # 向量化 运算
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) /len(y)

        def gradient_descent(X_b,y,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
            """梯度下降法"""
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
        X_b = np.hstack([ np.ones((len(X_train),1)),X_train ])
        # 初始化 theta 为 0
        initial_theta = np.zeros(X_b.shape[1])
        # eta 学习率
        self._theta = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self    

    def predict_proba(self,X_predict):
        """[预测概率]给定等待预测数据集X_predict,返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None,\
        "must fit before predict"

        assert X_predict.shape[1] == len(self.coef_) ,\
        "the featurn number of X_predict must be equal to X_train"
        # 预测值
        X_b = np.hstack([np.ones((len(X_predict),1)),X_predict])
        return  self._sigmoid(X_b.dot(self._theta))


    def predict(self,X_predict):
        """给定等待cbij数据集X_predict,返回表示X_predict的结果向量"""

        assert self.intercept_ is not None and self.coef_ is not None,\
        "must fit before predict"

        assert X_predict.shape[1] == len(self.coef_) ,\
        "the featurn number of X_predict must be equal to X_train"

        # 预测值
        proba = self.predict_proba(X_predict)

        return  np.array(proba >= 0.5, dtype='int')

    def score(self,X_test,y_test):
        """根据预测数据集X_test,y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return  accuracy_score(y_test,y_predict)

    def __repr__(self):
        return "LogisticRegression()"
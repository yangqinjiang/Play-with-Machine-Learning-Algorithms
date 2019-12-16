import  numpy as np
from .metrics import  r2_score
#  简单的线性回归模型,foreach计算
class SimpleLinearRegression1:
    def __init__(self):
        """初始化Simple Linear Regression模型"""
        # 对应的函数: y = a*x + b
        self.a_ = None
        self.b_ = None


    def fit(self,x_train,y_train):
         """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
         # 本类,只能实现简单线性回归,即维度是1的向量
         assert x_train.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."

         # 两个训练数据集的大小必须一致
         assert len(x_train) == len(y_train),\
            "the size of x_train must be equal to the size of y_train"

         #计算平均值
         x_mean = np.mean(x_train)
         y_mean = np.mean(y_train)

         # foreach 计算分子,分母
         num = 0.0
         d = 0.0
         for x,y in zip(x_train,y_train):
             num += ( x-x_mean) * (y - y_mean)
             d += (x - x_mean) ** 2

         self.a_ = num /d
         self.b_ = y_mean - self.a_ * x_mean

         return self

    def predict(self,x_predict):
         """给定等待预测数据集x_predict,返回表示x_predict的结果向量"""
         # 本类,只能实现简单线性回归,即维度是1的向量
         assert x_predict.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."

         assert self.a_ is not  None and self.b_ is not None,\
            "must fit before predict"

         return np.array([self._predict(x) for x in x_predict])


    def _predict(self,x_single):
         """给定单个等待预测数据x,返回x的预测结题值"""

         # 使用简单线性回归的公式
         return  self.a_ * x_single + self.b_

    def score(self,x_test,y_test):
        """根据预测数据集 x_test 和y_test确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return  r2_score(y_test,y_predict)

    def __repr__(self):
         return  "SimpleLinearRegression1()"


#  简单的线性回归模型,向量化计算
class SimpleLinearRegression2:
    def __init__(self):
        """初始化Simple Linear Regression模型"""
        # 对应的函数: y = a*x + b
        self.a_ = None
        self.b_ = None


    def fit(self,x_train,y_train):
         """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
         # 本类,只能实现简单线性回归,即维度是1的向量
         assert x_train.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."

         # 两个训练数据集的大小必须一致
         assert len(x_train) == len(y_train),\
            "the size of x_train must be equal to the size of y_train"

         #计算平均值
         x_mean = np.mean(x_train)
         y_mean = np.mean(y_train)

         # 向量化计算 a_,

         self.a_ = (x_train - x_mean).dot(y_train-y_mean) / ( x_train - x_mean).dot(x_train - x_mean)
         # b_ 的计算方式不变
         self.b_ = y_mean - self.a_ * x_mean

         return self

    def predict(self,x_predict):
         """给定等待预测数据集x_predict,返回表示x_predict的结果向量"""
         # 本类,只能实现简单线性回归,即维度是1的向量
         assert x_predict.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."

         assert self.a_ is not  None and self.b_ is not None,\
            "must fit before predict"

         return np.array([self._predict(x) for x in x_predict])


    def _predict(self,x_single):
         """给定单个等待预测数据x,返回x的预测结题值"""

         # 使用简单线性回归的公式
         return  self.a_ * x_single + self.b_

    def score(self,x_test,y_test):
        """根据预测数据集 x_test 和y_test确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return  r2_score(y_test,y_predict)

    def __repr__(self):
         return  "SimpleLinearRegression2()"
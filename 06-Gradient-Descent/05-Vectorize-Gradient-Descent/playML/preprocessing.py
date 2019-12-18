import numpy as np

# 仅能处理二维的矩阵
# StandardScaler 作用：去均值和方差归一化。
# 且是针对每一个特征维度来做的，而不是针对样本。 
# 【注：】 并不是所有的标准化都能给estimator带来好处
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None  # 标准差

    def fit(self,X):
        """根据训练数据集X获取数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2"
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

    def transform(self,X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform!"

        assert X.shape[1] == len(self.mean_) ,\
            "the feature number of X must be equal to mean_ and std_"

        # 创建一个与X相同的shape
        resX = np.empty(shape=X.shape,dtype=float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]

        return resX
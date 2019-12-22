import numpy as np

class PCA:
    def __init__(self,n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None
    
    def fit(self,X,eta=0.01,n_iters = 1e4):
        """获得数据集X的前n个主成分"""

        # 主成分的个数 <= 训练数据集的特征值
        assert self.n_components <= X.shape[1],\
            "n_components must not be greater than the featurn number of X"

        def demean(X):
            return X - np.mean(X,axis=0)

        def f(w,X):
            """目标函数: 求w, 使得函数f(x) 最大,  见上图 """
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w,X):
            """求导: 数学的形式"""
            return X.T.dot(X.dot(w)) * 2. / len(X)


        def direction(w):
            """有方向的单位向量"""
            return w / np.linalg.norm(w)

        def first_component(X,initial_w,eta,n_iters=1e4,epsilon=1e-8):
            """第一主成分,使用梯度上升法求解"""
            w = direction(initial_w) # 转化成单位方向
            cur_iter = 0 # 当前循环计数
            
            while cur_iter < n_iters:
                gradient = df(w,X) # 1,求梯度
                last_w = w # 记录上一次的w
                w = w + eta * gradient # 使用梯度计算新的w
                w = direction(w) # 注意1,每次求一个单位方向 向量, 优化性能?
                if (abs( f(w,X) - f(last_w,X) ) < epsilon):
                    break
                
                cur_iter += 1
            
            return w
        
        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components,X.shape[1]))
        for i in range(self.n_components):
            # 随机生成 w
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca,initial_w,eta,n_iters)
            self.components_[i,:] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1,1) * w

        return self

    def transform(self,X):
        """将给定的X,映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self,X):
        """将给定的X,反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
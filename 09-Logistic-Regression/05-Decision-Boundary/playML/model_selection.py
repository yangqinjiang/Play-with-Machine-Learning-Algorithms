import numpy as np

def train_test_split(X,y,test_ratio=0.2,seed=None):
    """将数据X和y按照test_train分割成X_train,X_test,y_train,y_test"""
    assert X.shape[0] == y.shape[0],\
        "the size of X must be equal to the size of y"

    assert 0.0 <= test_ratio <= 1.0,\
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    # 随机打乱数据
    shuffled_indexes = np.random.permutation(len(X)) 
    #测试数据的比例 即 80% 用于训练, 20%用于测试数据
    test_size = int(len(X) * test_ratio)
    # 获取随机的索引
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:] 
    # 获取真实的随机值

    # 训练集
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    # 测试集
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    return X_train,X_test,y_train,y_test
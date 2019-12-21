### 多项式回归:对非线性的数据进行拟合 , 不过,拟合过程也有陷阱的
如果原始数据是使用二次方程生成的,
### 欠拟合 underfitting:
- 那么使用一次方程得到的拟合结果,显然是欠拟合的.
- 算法所训练的模型不能完整表达数据关系
![avatar](images/2.png)
### 过拟合
- 算法所训练的模型过多地表达数据间的 '噪音' 关系, 但噪音数据,并不是我们想要的

- 如果是使用高于二次方程得到的拟合结果, 尤其是degree=10,100 ,甚至更高的degree进行拟合的话,结果一定是过拟合
![avatar](images/1.png)


### 学习曲线
随着训练样本的增多,算法训练出的模型的表现能力  

![avatar](images/3.png)
![avatar](images/4.png)


### Validation 和Cross Validation( 验证数据集与交叉验证)
![avatar](images/5.png)
![avatar](images/6.png)
 ### 模型泛化
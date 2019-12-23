# 分类准确度的问题
对于极度偏斜(Skewed Data)的数据, 只使用分类准确度是远远不够的
极度偏斜: 是指用于分类的数据有的很多,  有点很少. 例如癌症患者的数据,发病率0.01%, 健康人数99.99%
 必须引用其他的算法指标: 精准率和召回率, F1 Score
 - 准确度不能很好的表达模型的效果
![avatar](images/4.png)
![avatar](images/2.png)
![avatar](images/3.png)
# 混淆矩阵
![avatar](images/1.png)

# 精准率和召回率
- 精准率 :在预测数据为1的集合,那么预测正确的概率是多少
![avatar](images/5.png)
- 召回率: 在真实数据中, 预测正确的概率是多少
![avatar](images/6.png)

![avatar](images/7.png)

# 无意义的预测模型
- 因为精准率和召回率为0或无意义
![avatar](images/8.png)
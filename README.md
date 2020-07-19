# DeepInf

## 实现细节

### 参考
阅读论文的过程中结合论文提供的源码进行，因此在自行实现的过程中，有些思路参考自其源码，但模型参数均凭自行训练及经验推算得到，原论文有一定差距。

### 试错
由于受环境算力所限，无法进行更多的参数迭代（GAT一次完整训练耗时数十小时），模型仅为当前局部最优。

### 改进
尝试使用集成学习算法对模型进行改进，但训练时间不足，并没有得到满意的结果，因此没有体现在提交的代码上。

## 预测结果
loss: 0.5707 AUC: 0.7694 Prec: 0.4440 Rec: 0.6611 F1: 0.5312
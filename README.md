# FedMD_Pytorch

## Paper

FedMD: Heterogeneous Federated Learning via Model Distillation. 
Preprint on https://arxiv.org/abs/1910.03581.

## Environment

- GeForce GTX 1050 Ti
- Python 3.7.6
- torch 1.6.0
- torchvision 0.7.0

## Paper Abstract

Transfering learning is another  major framework addressing the scarcity of private data. In this work, our private datasets can be as small as a few samples per class. Threrefore using transfer learning from a large public dataet is imperative in addition to federated learning. We leverage the power of transfer learning in two ways. First, before entering the collaboration, each model is fully trained first on the public data and then on its own private data. Second, and more importantly, the blackbox models communicate based on their output class scores on samples from the public dataset. This is realized through knowledge distillation, which has been capable of transmitting learned information in a model agnostic way.

**迁移学习是解决私有数据稀缺的另一个主要框架**。 在这项工作中，我们的**私有数据集每个类可能只有几个样本**。 除了使用联合学习之外，还必须从大型公共数据集中使用转移学习。 我们以两种方式利用转移学习的力量。 首先，在进入协作之前，**首先充分利用公共数据，然后再利用其自己的私有数据来充分利用每种模型**。 其次，更重要的是，黑盒模型基于它们在公共数据集中样本上的输出类别分数来进行通信。 这是通过**知识蒸馏**实现的，它已经能够以模型不可知的方式传输学习到的信息。

We implement such a communication protocol by leveraging the power of transfer learning and knowledge distillation.



Transfer learning: Before a participant starts the collaboration phase, its model must first undergo the entire transfer learning process. It will be trained fully on the public dataset and then on its own private data. Therefore any future improvements are compared to this baseline.

转移学习：参与者开始协作阶段之前，其模型必须首先经历整个转移学习过程。 **将在公共数据集上对其进行全面培训，然后在其自己的私有数据上进行培训**。 因此，将将来的任何改进都与此基准进行比较。

交流：在这里 重新定义了公有数据集的作用，作为basis of communication between models，通过使用知识蒸馏。每个参与者的模型通过共享class scores 表达自己的模型能力，然后考虑到使用完整的lalrge public数据集导致巨大的交流压力。实际上，每次选择一个小很多的公有数据集的子集，这样cost 是可控的，does not scale with the complexity of participating models.

## 问题定义

m个参与者

每个local 有一个非常小的labeled dataset $D_k$ :={($x_i^k$,$y_i$)}^{N_k}_{i=1}（可能是或者不是同分布）

还有一个大的公有数据集 $D_0$ :={($x_i^0$,$y_i$)}^{N_0}_{i=1} 每个人都可以access

大家独立设计自己的模型$f_k$

目的是建立一个合作网络提升$f_k$性能（强于自己在自己的$D_k$和$D_0$上训练）

**FedMD**

Input：公有数据集$D_0$ 私有数据集$D_k$ 独立设计的模型$f_k$

Output：训练后的模型$f_k$

迁移学习：每方local的$f_k$先收敛在公有的$D_0$，再在私有数据集$D_k$

for j=1,2,....P do

交流：每个模型计算class score $f_k$(x_i^0) on 公有数据集，然后transmits 结果到中央服务器

整合：把大家的结果直接取平均	

Distribute: 每个参与者下载更新后的结果

Digest：参与者训练fk模型 去接近 在公有数据集上大家的平均结果

Revist：每个模型在自己的私有数据集上再训练几个周期

end


























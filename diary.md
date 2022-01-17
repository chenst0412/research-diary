# **2022.01.13**

## Exemplar-based Layout Fine-tuning for Node-link Diagrams

步骤:

* 计算整个图的node embedding来检索相似子结构，给图初始布局
* 用户从全图中指定一个exempler，通过某种算法检索出拓扑结构类似于exempler的子结构。用户也可以直接在图中指定这些子结构子结构
* 用户修改exempler的布局，算法将该修改传播到所有前一步中的target子结构
* 算法将对子结构布局的修改合并到原图中，同构全局优化来平滑边界

### 1. WorkFlow

#### 1.1 node-embedding-based representation

通过node embedding将子图检索问题简化为一个多维数据的搜索问题。使用GraphWave用作node embedding

#### 1.2 Specifying Exempler and Targets

使用套索来指定一个子结构(Fig.3a)，并通过一种structure-query技术来检索和目标样例相似的子结构(参考文献[4]，CGH2019)。

#### 1.3 User-driven fine-tuning

dragging interaction来交互式地调整exempler布局，随后修改被转移到所有其他的目标子结构

#### 1.4 Global Layout Optimization

直接将修改后的子图布局嵌入到原图中会产生突兀的boundary，因而需要global optimization。

类似于stress majorization， 最小化每个节点对之间的位置的期望距离和实际距离。但是优化整个图需要较大的计算量，因此只需要优化target substructures周围的局部的图即可。定义为距离某个substructure小于d的所有节点的导出子图

### 2. Modification Transfer

source substructure$S=(V^S, E^S)$, target substructure$T=(V^T, E^T)$ 三个步骤:

* Marker selection: 使用图匹配方法对齐S和T，找到匹配度高的节点
* 改变T->T‘来模拟S，并扩展M
* 改变T’来模拟S‘

### 3. Marker Selection

预先对exempler和target substructure单独进行布局，然后使用Factorized Graph Matching（FGMU）进行图匹配。但是算法会对所有的点都进行匹配，从而产生一些不是那么好的配对点对。因此需要进行filtering. 首先依据配对来对两个layout进行对齐(Section4.2)，然后找到fine correspondences $(c_i^s, c_i^t)$:


* 两者之间的距离小于它们相邻边的平均距离乘以某个ratio
* $c_i^s$的邻居节点尽可能和$c_i^t$的邻居节点配对

### 4. Layout Simulation

#### 4.1 Aligning

对目标子结构进行放缩，旋转，平移来最小化两个布局之间的差异，因为二者期望是类似的子结构，拥有者类似的布局位置。

需要预定义一小部分markers $M=\{(m_i^s, m_i^t)\}$, 通过仿射变换$R$，解下面的最优化问题

$$\min_R \sum^{|M|}\|Rm_i^t-m_i^s\|^2$$

#### 4.2 Deforming

改变T的结构来模拟S，即让所有的$m_i^t$的位置接近于$m_i^s$,定义energy function

$$E = E_S + \gamma E_M$$
其中$E_M$是所有markers对的距离之和
$E_S$衡量了是T变换到T’的layout的变化

$E_S=\alpha E_O + \beta E_D$
其中$E_O$保持所有节点对在变换layout后的大致方向，$E_D$保持变换后所有节点对之间的大致距离

然后使用类似于stress majorization的优化方法来优化$E_O$和$E_D$

#### 4.3 Matching

在进行Deforming之后，再一次进行节点之间的匹配。所有距离在$r_j$之内的来自S和T的节点对作为一个匹配对应的候选，使用匈牙利算法来找到这些配对之间的一个最大数量的配对

Aligning, Deforming, Matching交替进行，来将目标的布局尽量接近于给定的样例的布局变化

## **A Visual Analytics Framework for Contrastive Network Analysis**

1. ### **Contrative network representation learning**

   使用DeepGL做表征学习(NRL)提取特征，使用cPCA做对比学习(CL)

#### **1.1 NRL with DeepGL**

生成的特征由base feature$X$和relational function $f$组成。base feature衡量了能从每个节点得到的特征，如入度、出度或一些更深层的属性等等。relational function是一个关系特征操作子（RFO）的组合，每个特征operator总结了一个节点和它的one-hop节点，也就是它的邻居节点之间的某种运算关系，如mean，sum，maximum。

#### **1.2 CL with cPCA**

cPCA最大化目标$X_t$投影后的方差而最小化background方差X_B，是PCA的扩展. 得到投影后的表征$Y_T,Y_B$.

1. ### **Design Consideration**

-  Disovery

- Interpretability

-  Intuitiveness

-  Flexibility

1. ### **Visualization**

contrastive network比对流程:

- DeepGL生成网络特征

- cPCA生成target和background对比的网络表征

-  确定target network是否有独特性(view a)

- 解释网络特征和对比主成分

- 关联独特性和补充的可视化

界面:

​                 ![img](figs/2020-01-13-fig1.png)        

-  A: target network和background network在cPCA后的特征分布

- B: feature中不同的特征对于主成分方向的贡献程度

- C: target network和background network中的在view B中选择的feature的概率分布，其实也就是特征的频率分布

-  D, E: 具体的target network和background network



在cPCA后投影的低维空间内，target network的独特性将被放大，表现为其特征点的空间位置更加分散(方差更大)，也代表着两个图之间的属性差异。

## **备注**

今天的两篇文章是之前积累但未详细阅读的两篇工作。周计划从1.14开始

## **周计划 2022.01.14~2022.01.20**

## **本周粗计划细读文章六篇如下：**

- 2014_Deepwalk online learning of social representation
- 2016_node2vec scalable feature learning for networks
- 2017_struc2vec learning node representations from structural identity
- 2019_motif2vec_Motif_Aware_Node_Representation_Learning_for_Heterogeneous_Networks
- 2015_Distilling the knowledge in a Neural Network
- 2017_VIGOR_Interactive_Visual_Exploration_of_Graph_Query_Results

部分文章可能能较快阅读完成，或是某些文章的阅读可能涉及到学习额外内容，因此只是大致计划，具体阅读以实际为准。

【Dong】文中提到文献4是指哪篇文章？开了好头，继续加油！可以在阅读时记录下自己认为可以改进的点。



# 2022.01.14

## 2016_node2vec scalable feature learning for networks

学习网络节点的低维空间映射，最大化保持网络节点邻居的likelihood

presented work: 提出node2vec，一个网络特征学习的半监督算法，使用SGD优化，返回在d维空间下的特征表征来最大化保持节点邻居的likelihood，使用2阶随机游走来生成网络节点的邻居

本文贡献

* node2vec算法
* 证明了node2vec是符合建立的网络科学原理的，提供在寻找不同等价物表征时的灵活性
* 基于neiborhood preserving objectives扩展node2vec和其他特征学习算法
* 经验性地在现实数据集上测试了node2vec

### 1. Feature Learning Framework

将skip-gram架构扩展到网络中，致力于优化下面的目标函数
$$
max_f \sum_{u\in V}\log Pr(N_s(u)|f(u))
$$
$N_s$是节点邻居，$f$是节点特征表征

两个假设：

* 条件独立：给定特征表征，观测一个邻居节点和观测其他邻居节点是独立的，即
  $$
  Pr(N_s(u)|f(u))=\Pi_{n_i\in N_s(u)}Pr(n_i|f(u))
  $$

* 源节点和邻居节点在特征空间对彼此有着对称的影响
  $$
  Pr(n_i|f(u))=\frac{\exp(f(n_i)\cdot f(u))}{\sum_{v\in V}\exp (f(v)\cdot f(u))}
  $$
  （由于标准化向量的内积通常对应着在向量空间二者的角度关系，因此最大化上面的似然函数，相当于使得角度空间中两个结点更近，对应在原图上就是邻居节点）

从而目标函数变为
$$
\max_f\sum[-\log Z_u+\sum_{n_i\in N_s(u)}f(n_i)\cdot f(u)]
$$
$Z_u$难于计算，因此采用负采样(negtive sampling)的方法来近似。使用随机梯度下降来优化目标函数

#### 1.1 Classic search strategy

BFS和DFS。

图上节点的两种相似性：homophily和structural equivalence

#### 1.2 Node2vec

可视作介于dfs和bfs的方法

##### 1.2.1 random walks

依照一定的转移概率从起始节点出发生成一条random walk

##### 1.2.2 search bias $\alpha$

定义了参数$p$和参数$q$来控制random walk的过程。假设当前的random walk刚刚经过边$(t, v)$，现在要决定下一步$(v, x)$，设定转移的unnormalized转移概率$\pi_{vx}=\alpha_{pq}(t,x)\cdot w_{vx}$, 其中$\alpha_{pq}(t, x)$为

* $\frac{1}{p}, d_{tx}=0$
* $1, d_{tx}=1$
* $\frac{1}{q}, d_{tx}=2$

$d_{tx}$指两个节点$t, x$的图上距离

random walk的好处：

* 便于计算，空间复杂度低
* 时间复杂度相比较于classic search也很低

##### 1.2.3 算法

random walk存在隐形bias，源自于选择的起始节点$u$。但是任务是学习所有节点的向量表征，因此算法从每个节点起始都会做random walk。



# 2022.01.15

## 2014_Deepwalk online learning of social representation

本文贡献

* 引入了deep learning来作为图分析工具，提取鲁棒的合适作为模型输入的表征。DEEPWALK学习短random walk的结构规律性
* 在多分类任务上测试了我们的特征提取方法
* 证明了算法的可扩展性，并提供了微小修改使其可以作为流水线处理的一部分。

### 1. Learning Social Representations

学到的表征应该有如下特征

* Adaptability: 真实社交网络会不断变化，新的社交关系不应该要求重复训练
* Community aware：在latent space两个节点的距离应该反应节点成员在社交网络上的相似性（homophily）

* Low dimensional
* Continous: 低维表征是连续性的，有着平滑的决策边界

#### 1.1 Random Walk

除了捕捉社区信息，使用random walk作为算法基础还有两个好处

* 局部exploration方便并行化
* 依赖short random walk获取的信息使得在适应图结构有微小变化的时候不需要重新进行全局的计算成为可能。可以迭代通过新的random walk来更新模型。

#### 1.2 Connection：Power Laws

如果连接图节点度的分布遵循幂定律，那么图中哪些节点出现在short random walk中也遵从幂定律分布。幂定律分布，例如zipf law，zipf law可以表述为在自然语言语料库中，一个单词出现的频率和他在频率表中的排名成反比，频率最高的单词频率大概是第二名的两倍，第二名是第四名的两倍等等。

#### 1.3 语言模型

给定词序列$W_1^n=(w_0, w_1, ..., w_n)$，通常希望在训练语料库中最大化$\Pr(w_n|w_0, w_1, ..., w_{n-1})$，在Deep Walk中也一样，最大化$\Pr(v_i|v_0, ..., v_{i-1})$。由于目标是学习latent representation而不是节点的co-occurence，因此引入映射函数$\Phi\in V^{|V|\times d}$，从而目标计算$\Pr(v_i|(\Phi(v_1),...\Phi(v_{i-1})))$，然而随着序列的增长，这种计算很困难。自然语言处理中对此问题有三个relaxation：

* 使用一个单词来预测上下文
* 上下文由给定单词的前后组成
* 移除了序列中的顺序限制，一个窗口内的单词顺序不考虑

从而优化问题变成了
$$
\min_\Phi -\log \Pr (\{v_{i-w}, ..., v_{i+w}\} \setminus v_i | Phi(v_i))
$$
其中$w$是窗口。概率的意思是给定顶点$v_i$时，其$w$窗口范围内各顶点的似然概率分布

### 2. Method

类比自语言模型

#### 2.1 random walk

算法由两部分组成：random walk generator和update procedure。

random walk generator：随机选取节点作为根节点，从根节点开始不断的随机选择邻居节点，直到达到规定长度。

update procedure：借鉴的是词向量模型中的skip-gram的思路，取得random walk之后，最大化滑动窗口中每个词在窗口中出现的概率

##### 2.1.1 Skip-Gram

语言模型中的skip-gram最大化出现在同一窗口的单词的cooccurance probability。在本文Deep Walk算法中，每取得一个random walk序列后，依据预定义的窗口大小$w$，对每个窗口$W_{v_i}[j-w, j+w]$内的节点$u_k$，计算$J(\Phi)=-\log \Pr(u_k|\Phi(v_j))$和$\Phi=\Phi - \alpha \cdot \frac{\partial J}{\partial \Phi}$，$\Phi$为节点的表征映射矩阵

#### 2.2 并行化

节点在random walk中出现的频率分布遵从于power law，这造成了不频繁单词的long tail现象，因此影响$\Phi$的更新是稀疏的，这使得可以使用随机梯度下降的异步版本，从而实现并行化更新。

### 3. 总结

文章是将自然语言处理中序列模型借鉴过来用网络节点来表示，node2vec借鉴了这篇文章的思路方法



# 2022.01.16

## 2017_struc2vec learning node representations from structural identity

本文提出struc2vec，一种新颖的学习节点structural identity的表征的灵活框架，其使用层级来度量节点在不同尺度下的相似度，构建多层图来编码结构相似性并生成节点的结构上下文。

通常最常用的确定节点的结构同一性的方法是基于distances或者recursions

* distances：利用节点邻居信息的距离函数用于度量每一对节点之间的距离，然后实行聚类和匹配来将节点放置在等价类上。
* recursions：构建相对于节点的递归并迭代展开直到收敛，最后的值决定等价类。

Deepwalk和node2vec在分类任务中成功但是在结构等价类任务中经常失败的原因，是很多现实网络中的节点特征呈现很强的homophily，这意味着在这些算法中相邻节点更可能有着相同的特征，而网络中距离较远的节点则往往被隔离开不被考虑，即不能很好的抓住相距较远的节点的structural equivalence。（但是node2vec中明明有说提出的random walk构造方法通过超参数平衡homophily和structural equivalence，这个说法略奇怪）

主要思路

* 评估了独立于节点和边的属性以及他们在网络中的位置的结构相似性。不要求连通图
* 建立层级衡量结构相似性
* 生成节点的随机上下文，这是通过在多层图上进行加权随机游走得到的结构相似节点序列，这种随机上下文可以通过语言模型来进行学习

结果标明Deepwalk和node2vec在捕捉structural identity上比较失败，但是该方法在此任务上执行的很好。

### 1. Struct2Vec

一个成功的方法应当有如下两个特性

* 节点隐性表征的距离应当和structural identity高度相关。
* 节点的隐性表征不应当依赖于节点和边的属性，以及他们的标签

步骤：

1. 对不同的邻居规模，确定每一对节点之间的结构相似性，从中可以构建出一个衡量节点结构相似度的层级，提供更多的信息来衡量每一个层级的节点相似度。
2. 构建加权多层图，所有节点都会在每一层出现，并且各层都对应于衡量结构相似性层级的一个level
3. 使用多层图来生成节点上下文，特别的，使用多层图上有偏随机游走来生成节点序列，序列更可能包含的是更有结构相似性的节点
4. 使用Skip-gram等语言模型来学习隐形表征

#### 1.1 Measuring structural similarity

不使用节点和边的属性来构建节点之间的结构相似性。

令$R_k(u)$表示节点$u$的图上距离为$k$的邻居，$s(S)$表示节点集合$S\subset V$的有序的度序列。对于两个节点$u, v$，定义$f_k(u, v)$表示当考虑$k$-hop邻居时两个节点的结构距离structural distance，**所谓$k$-hop指的是所有图上距离小于等于$k$的节点和他们之间所有的边**
$$
f_k(u, v)=f_{k-1}(u, v)+g(s(R_k(u)), s(R_k(v))), k\geq 0 \quad \&\& \quad |R_k(u)|, |R_k(v)| > 0
$$
其中$g(D_1, D_2)\geq0$衡量两个有序度序列的距离，且定义$f_{-1}=0$

然后需要定义衡量两个有序度序列的距离的方法。（$g$）。两个度序列可能有着不同的长度，且其中每个元素都在$[0, n-1]$之间且可能有重复。采用**Dynamic Time Wrapping(DTW)**的方法，DTW寻找两个序列最佳的alignment，给定距离函数$d(a, b)$，DTW将每个$a\in A$匹配到$b\in B$，使得所有匹配比较的节点对距离之和最小，本文采用的距离函数如下
$$
d(a, b)=\frac{\max(a, b)}{\min(a, b)}-1
$$
这样两个相同的序列的距离也就为0

#### 1.2 Constructing the context graph

令$M$表示multilayer graph，其中layer $k$通过节点的$k$-hop邻居来定义。$k^*$是图直径。每一层$k=0, ..., k^*$通过节点集的加权无向完全图来定义，两个节点之间的权值为
$$
w_k(u, v)=e^{-f_k(u, v)}
$$
这样两个节点之间的结构距离越小，他们之间边的权值越大，在每一层上都是这样，两个节点的结构相似度越大（基于不同的$k$），他们之间边的权值越大。

不同layer之间使用有向边连接。每个节点和他在其他$k+1, k-1$的layer的他自身有边的连接
$$
w(u_k, u_{k+1})=\log(\Gamma_k(u)+e), k=0, ..., k^*-1 \\
w(u_k, u_{k-1})=1, k=1, ..., k^*
$$
其中
$$
\Gamma_k(u)=\sum_{v\in V}\mathbf 1(w_k(u, v)>\bar{w_k})
$$
$\bar{w_k}$是完全图layer $k$中所有边权值的平均值，$\Gamma_k(u)$指的是与$u$相连的权值大于图权值平均值的边的数目，这相当于是度量节点$u$和图中其他节点之间的相似度，因为两对节点的边权值越大表示他们的结构相似度越大。

#### 1.3 Generating context for nodes

每一步开始前，random walk确定是改变layer还是walk在当前的layer，这里定义一个概率$q$来表示这个值。若在当前layer进行下一步，则下一步游走的概率如下
$$
p_k(u, v)=\frac{e^{-f_k(u, v)}}{Z_k(u)}
$$
这样random walk更倾向于structural similarity更高的节点。这样节点$u\in V$的上下文更倾向于是结构相似节点，而和他们的标签以及在原图中的位置无关。若选择改变layer，则概率为
$$
p_k(u_k, u_{k+1}) =\frac{w(u_k, u_{k+1})}{w(u_k, u_{k+1})+w(u_k, u_{k-1})} \\
p_k(u_k, u_{k-1}) =1-p_k(u_k, u_{k+1})
$$
即还是由概率来决定，从直觉上理解，当前$k$值下和节点$u$相似的节点较多，则倾向于扩大范围，到$k+1$层游走

#### 1.4 Learning a language model

和deepwalk，node2vec一样，构造随机游走后借鉴自然语言处理方面的方法，学习节点laten representation



# 2022.01.17

## 2019_motif2vec_Motif_Aware_Node_Representation_Learning_for_Heterogeneous_Networks


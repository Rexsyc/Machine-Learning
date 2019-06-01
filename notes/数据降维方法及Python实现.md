# 数据降维方法及Python实现

##一、数据降维了解

**1.1、数据降维原理：**机器学习领域中所谓的降维就是指采用某种映射方法，将原高维空间中的数据点映射到低维度的空间中。降维的本质是学习一个映射函数 f : x->y，其中x是原始数据点的表达，目前最多使用向量表达形式。 y是数据点映射后的低维向量表达，通常y的维度小于x的维度（当然提高维度也是可以的）。f可能是显式的或隐式的、线性的或非线性的；

**1.2、不进行数据降维的可能的影响：**数据集包含过多的数据噪声时，会导致算法的性能达不到预期。移除信息量较少甚至无效信息维度可能会帮助我们构建更具扩展性、通用性的数据模型。该数据模型在新数据集上的表现可能会更好；

**1.3、数据降维的原则：**在减少数据列数的同时保证丢失的数据信息尽可能少。

##二、数据降维作用

* 降低时间复杂度和空间复杂度；

* 节省了提取不必要特征的花销；

* 去掉数据集中夹杂着的噪声数据；

* 较简单的模型在小数据集有更强的鲁棒性；

* 当数据能用较少的特征进行解释，我们可以更好的解释数据，使得我们可以提取知识；

* 实现数据可视化。

  

##三、数据降维方法

**3.1、缺失值比率 (Missing Values Ratio)：**该方法的是基于包含太多缺失值的数据列包含有用信息的可能性较少。因此，可以将数据列缺失值大于某个阈值的列去掉。阈值越高，降维方法更为积极，即降维越少。

```python
## 代码示例（缺失值比率）
 
def na_count(data):
    '''各指标缺失规模及缺失占比统计 '''
    data_count = data.count()
    na_count = len(data) - data_count
    na_rate = na_count/len(data)
    result = pd.concat([data_count,na_count,na_rate],axis = 1)
    return result
 
def miss_data_handle(data):
    '''高缺失字段处理'''
    table_col = data.columns
    table_col_list = table_col.values.tolist()          
    row_length = len(data)
    for col_key in table_col_list:
        non_sum1 = data[col_key].isnull().sum()
        if non_sum1/row_length >= 0.8:
            data[col_key] = data[col_key].dropna(axis = 1) 
    return data
```

**3.2、随机森林/组合树 (Random Forests)：**组合决策树通常又被成为随机森林，它在进行特征选择与构建有效的分类器时非常有用。一种常用的降维方法是对目标属性产生许多巨大的树，然后根据对每个属性的统计结果找到信息量最大的特征子集。例如，我们能够对一个非常巨大的数据集生成层次非常浅的树，每颗树只训练一小部分属性。如果一个属性经常成为最佳分裂属性，那么它很有可能是需要保留的信息特征。对随机森林数据属性的统计评分会向我们揭示与其它属性相比，哪个属性才是预测能力最好的属性。

```python
## 代码示例（随机森林/组合树）
 
target_col = 'IS_SUCCESS'   #响应变量
ipt_col = list(data.columns)  #数据集维度列表
 
def data_sample(data, col=target_col, smp=3):
    data_1 = data[data[col] == 1].sample(frac=1)
    data_0 = data[data[col] == 0].sample(n=len(data_1)*smp)
    data = pd.concat([data_1, data_0]).reset_index()
    return data
 
def train_test_spl(data):
        '''数据切分'''
        X_train, X_test, y_train, y_test = train_test_split(
            data[ipt_col], data[target_col], test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
 
def feture_extracted(data):
    global ipt_col
    ipt_col= list(data.columns)
    ipt_col.remove(target_col)
    sample_present = [1, 2, 4, 6, 8, 10, 12, 15]   # 定义抽样比例
    alpha = 0.9
    f1_score_list = []
    model_dict = {}
    for i in sample_present:
        try:
            data = data_sample(data, col=target_col, smp=i)
        except ValueError:
            break
        X_train, X_test, y_train, y_test = train_test_spl(data)  
        model = RandomForestClassifier()
        model = model.fit(X_train, y_train)
        model_pred = model.predict(X_test)
        f1_score = metrics.f1_score(y_test, model_pred)
        f1_score_list.append(f1_score)
        model_dict[i] = model
    max_f1_index = f1_score_list.index(max(f1_score_list))
    print('最优的抽样比例是：1:',sample_present[max_f1_index])
    d = dict(zip(ipt_col, [float('%.3f' %i) for i in model_dict[sample_present[max_f1_index]].feature_importances_]))
    f = zip(d.values(), d.keys())
    importance_df = pd.DataFrame(sorted(f, reverse=True), columns=['importance', 'feture_name'])
    list_imp = np.cumsum(importance_df['importance']).tolist()
    for i, j in enumerate(list_imp):
        if j >= alpha:
            break
    print('大于alpha的特征及重要性如下：\n',importance_df.iloc[0:i+1, :])
    print('其特征如下：')
    feture_selected = importance_df.iloc[0:i+1, 1].tolist()
    print(feture_selected)
    return feture_selected
```

**3.3、低方差滤波 (Low Variance Filter)：**与缺失值比率进行数据降维方法相似，该方法假设数据列变化非常小的列包含的信息量少。因此，所有的数据列方差小的列被移除。需要注意的一点是：方差与数据范围相关的，因此在采用该方法前需要对数据做归一化处理。 

```python
## 代码示例（低方差滤波）
 
var = data.var()
numeric = data.columns
variable = [ ]
for i in range(0,len(var)):
    if var[i]>= 10:   # 将阈值设置为10％
       variable.append(data[i+1])
```

**3.4、高相关滤波 (High Correlation Filter)：**高相关滤波认为当两列数据变化趋势相似时，它们包含的信息也相似。这样，使用相似列中的一列就可以满足机器学习模型。对于数值列之间的相似性通过计算相关系数来表示，对于数据类型为类别型的相关系数可以通过计算皮尔逊卡方值来表示。相关系数大于某个阈值的两列只保留一列。同样要注意的是：相关系数对范围敏感，所以在计算之前也需要对数据进行归一化处理。

```python
## 代码示例（高相关滤波）
 
k = 0.8  #配置参数
def data_corr_analysis(data, sigmod = k):
    '''相关性分析：返回出原始数据的相关性矩阵以及根据阈值筛选之后的相关性较高的变量'''
    corr_data = data.corr()         
    for i in range(len(corr_data)):
        for j in range(len(corr_data)):
            if j == i:
                corr_data.iloc[i, j] = 0
 
    x, y, corr_xishu = [], [], []
    for i in list(corr_data.index):         
        for j in list(corr_data.columns):  
            if abs(corr_data.loc[i, j]) > sigmod:        
                x.append(i)
                y.append(j)
                corr_xishu.append(corr_data.loc[i, j])
    z = [[x[i], y[i], corr_xishu[i]] for i in range(len(x))]
    high_corr = pd.DataFrame(z, columns=['VAR1','VAR2','CORR_XISHU'])
    return high_corr
```

**3.5、主成分分析 (PCA)：**主成分分析是一个统计过程，该过程通过正交变换将原始的 n 维数据集变换到一个新的被称做主成分的数据集中。变换后的结果中，第一个主成分具有最大的方差值，每个后续的成分在与前述主成分正交条件限制下具有最大方差。降维时仅保存前 m(m < n) 个主成分即可保持最大的数据信息量。

**3.5.1、主要思想：**把数据从原来的坐标系转换到新的坐标系，新坐标系的选择由数据本身决定：第一个新坐标轴选择的是原始数据中方差最大的方向，第二个新坐标轴选择和第一个坐标轴正交且具有方差次大的方向。此过程一直重复，重复次数为原始数据中特征的数目。大部分方差都集中在最前面的几个新坐标轴中。因此，可以忽略剩下的坐标轴，即对数据进行了降维处理。

**3.5.2、注意：**

* 主成分变换对正交向量的尺度敏感。数据在变换前需要进行归一化处理。
* 新的主成分并不是由实际系统产生的，因此在进行 PCA 变换后会丧失数据的解释性。如果说，数据的解释能力对你的分析来说很重要，那么 PCA 对你来说可能就不适用了。

```python
## 代码示例（PCA）
 
from sklearn.decomposition import PCA
pca = PCA()  
pca = PCA(n_components = None,copy = True,whiten = False)
pca.fit(data)
pca.components_ 
pca.explained_variance_ratio_ 
 
pca = PCA(3)  #观察主成分累计贡献率,重新建立PCA模型
pca.fit(data)
low_d = pca.transform(data) 
```



**3.6、因子分析（FA） ：**通过研究众多变量之间的内部依赖关系，探求观测数据中的基本结构，并用少数几个假想变量来表示其基本的数据结构。这几个假想变量能够反映原来众多变量的主要信息。原始的变量是可观测的显在变量，而假想变量是不可观测的潜在变量，称为因子。

​	因子分析又存在两个方向，一个是探索性因子分析。另一个是验证性因子分析。探索性因子分析是不确定一堆自变量背后有几个因子，我们通过这种方法试图寻找到这几个因子。而验证性因子分析是已经假设自变量背后有几个因子，试图通过这种方法去验证一下这种假设是否正确。验证性因子分析又和结构方程模型有很大关系。

```python
## 代码示例（FA）
 
import pandas as pd
import numpy as np
import math
df = pd.DataFrame(mydata)
 
#样本离差矩阵
mydata_mean = mydata.mean()
E = np.mat(np.zeros((14, 14)))
for i in range(len(mydata)):
    E += (mydata.iloc[i, :].reshape(14, 1) - mydata_mean.reshape(14, 1)) * (mydata.iloc[i, :].reshape(1, 14) - mydata_mean.reshape(1, 14))
    
#样本相关性矩阵
R = np.mat(np.zeros((14, 14)))
for i in range(14):
    for j in range(14):
        R[i, j] = E[i, j]/math.sqrt(E[i, i] * E[j, j])
        
import numpy.linalg as nlg
eig_value, eigvector = nlg.eig(R)
eig = pd.DataFrame()
eig['names'] = mydata.columns
eig['eig_value'] = eig_value
eig.sort_values('eig_value', ascending=False, inplace=True)
 
#求因子模型的因子载荷阵，寻找公共因子个数m
for m in range(1, 14):
    if eig['eig_value'][:m].sum()/eig['eig_value'].sum() >= 0.8:
        print(m)
        break
 
 #因子载荷矩阵
A  = np.mat(np.zeros((14, 6)))
for i in range(5):
    A[:,i]=math.sqrt(eig_value[i])*eigvector[:,i]
a=pd.DataFrame(A)
a.columns=['factor1','factor2','factor3','factor4','factor5','factor6']
```

**3.7、反向特征消除 (Backward Feature Elimination)：**在该方法中，所有分类算法先用 n 个特征进行训练。每次降维操作，采用 n-1 个特征对分类器训练 n 次，得到新的 n 个分类器。将新分类器中错分率变化最小的分类器所用的 n-1 维特征作为降维后的特征集。不断的对该过程进行迭代，即可得到降维后的结果。第k 次迭代过程中得到的是 n-k 维特征分类器。通过选择最大的错误容忍率，我们可以得到在选择分类器上达到指定分类性能最小需要多少个特征。

```python
## 代码示例（反向消除特征）
 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import datasets
 
df = data.drop('IS_SUCCESS', 1)
lreg = LinearRegression()
rfe = RFE(lreg, 10)
rfe = rfe.fit_transform(df, data.IS_SUCCESS)
```

**3.8、前向特征构造 (Forward Feature Construction)：**前向特征构建是反向特征消除的反过程。在前向特征过程中，我们从 1 个特征开始，每次训练添加一个让分类器性能提升最大的特征。前向特征构造和反向特征消除都十分耗时。它们通常用于输入维数已经相对较低的数据集。

```python
## 代码示例（前项特征构造）
 
from sklearn.feature_selection import f_regression
ffs = f_regression(df,data.IS_SUCCESS)
 
variable = [ ]
for i in range(0,len(df.columns)-1):
    if ffs[0][i] >=10:
       variable.append(df.columns[i])
```

##**数据降维除了上述提到的几种，还包括：**

* 随机投影(Random Projections)；
* 非负矩阵分解(N0n-negative Matrix Factorization)；
* 自动编码(Auto-encoders)；
* 卡方检测与信息增益(Chi-square and information gain)；
* 多维标定(Multidimensional Scaling)；
* 聚类(Clustering)以及贝叶斯模型(Bayesian Models)


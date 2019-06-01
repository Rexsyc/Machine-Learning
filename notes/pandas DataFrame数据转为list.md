# pandas DataFrame数据转为list

首先使用np.array()函数把DataFrame转化为np.ndarray()，再利用tolist()函数把np.ndarray()转为list，示例代码如下： 

```python
# -*- coding:utf-8-*-
import numpy as np
import pandas as pd

data_x = pd.read_csv("E:/Tianchi/result/features.csv",usecols=[2,3,4])#pd.dataframe
data_y =  pd.read_csv("E:/Tianchi/result/features.csv",usecols=[5])

train_data = np.array(data_x)#np.ndarray()
train_x_list=train_data.tolist()#list
print(train_x_list)
print(type(train_x_list))
```


# matlab中数组和矩阵的区别

数组中的元素可以是字符等
矩阵中的只能是数
这是二者最直观的区别。
因为矩阵是一个数学概念（线性代数里的），数组是个计算机上的概念。

《精通MATLAB6.5版》（张志涌编著，北京航空航天大学出版社）中说：

从外观形状和数据结构上看，二维数组和数学中的矩阵没有区别。但是矩阵作为一种变换或映射算子的体现，矩阵运算有着明确而严格的数学规则。而数组元算是Matlab软件所定义的规则，其目的是为了数据管理方便、操作简单、指令形式自然和执行计算的有效。虽然数组运算尚缺乏严谨的数学推理，虽然数组运算仍在完善和成熟中，但是它的作用和影响正随着matlab的发展而扩大。

###数组运算：

数与数组加减：k+/-A        %k加或减A的每个元素

数组乘数组：  A.*B         %对应元素相乘

数组乘方：　　A.^k         %A的每个元素k次方；k.^A，分别以k为底A的各元素为指数求幂值

数除以数组：  k./A和A./k   %k分别被A的元素除

数组除法：    左除A.\B右除B./A，对应元素相除

###矩阵运算：

数与矩阵加减：k+/-A             %等价于k*ones(size(A))+/-A

矩阵乘法：    A*B               %按数学定义的矩阵乘法规则

矩阵乘方：　　A^k               %k个矩阵A相乘

矩阵除法：    左除A\B右除B/A    %分别为AX=B和XA=B的解

可见，数组的运算很简单。若不考虑数学意义时，矩阵是数组的二维版本。

构造数组：

1、直接构造：用空格或逗号间隔数组元素

x=[1,2,3,4,5,6]

2、增量法构造：使用冒号操作符创建数组

a=first：end         %递增，且步长为1的数组

a=first:step:end     %指定增量步长值创建任何等差序列

3、用linspace函数构造

x=linspace（first，last，num）  %需要指定首尾值和元素总个数，步长根据num平均分配

构造矩阵

1、简单创建方法

用[]，逗号或空格格开各元素，分号隔开各行，注意各行具有相同的元素个数。

2、构造特殊矩阵

ones，zeros，eye，diag，magic，rand，randn，randpem

.....
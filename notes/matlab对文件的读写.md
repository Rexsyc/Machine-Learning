# matlab对文件的读写

## matlab读取文件函数总结

###1. load读取方式

* a．基本说明：

  只能读取数值数据，不能读取含文本的数据；日期按数值读取。

* b．调用方式：

  a=load(filename);

* c． 有无分隔符限制：

  无需输入分隔符，可自动识别空格、逗号、分号、制表符。

* d． 能否自定义读取范围：

  不能。

* e． 适用文件类型：

  txt、csv。

###2. importdata读取方式

* a．基本说明:

  可读取数值数据和含文本的数据，但是要求文本在数据的第一行或第一列。返回值分为数值部分（data）和文本部分（textdata）

* b．调用方式:

  a=importdata(filename,delm,nheaderlines);

  filename：文件名（如果文件在其他路径下，文件名前需加所在路径。）

  delm：分隔符

  nheaderlines:从第nheaderlines+1行开始读取数值数据。

* c．有无分隔符限制:

  多列数据时需输入分隔符。若不输入分隔符,整行会被作为字符串放入一列。

* d．能否自定义读取范围:

  可从某一行开始读取数值数据。若使用importdata按钮，则可自定义读取范围和设置数据类型。

* f． 适用文件类型

  txt、xls、xlsx、csv。

###3. textscan读取方式

* a．基本说明:

  可对列按照自定义格式读取数据，必须输入每列的读取格式，可跳过某个列或几列。按数值读取时，缺少值以NaN填补；按字符读取时，缺少值以空格填补。返回值按列放入元胞数组。

* b．调用方式:

  c =textscan(fid,'format',n,'param',value)

  fid：文件指针。使用textscan函数时需先使用fopen函数打开数据文件，返回给fid文件。若不再使用，则需用			fclose（fid）关闭文件。

  ‘format’：定义每列的读取格式。例如:%s表示按字符串读取、%d表示按整数读取、%D按日期读取、%*表示跳过该列。level%u8表示将level1读取成1，去掉level。

  ‘param’,value：这两个参数成对出现。例如’Delimiter’,’s’表示按分隔符为’,’进行读取。

* c．有无分隔符限制

  可自定义分隔符，不是必须的。

* d．能否自定义读取范围:

  可跳过某列或某几列数据，但要保证跳过的列是可读的，否则读取也会出错。

* e．适用文件类型

​        txt、csv

####4. textread读取方式

* a.基本说明:

  适用于格式统一的txt文件的一次性大批量读取。textread读取某个文件后，下次再用，textread读取这个文件时，还是会从文件头开始读取。

* b.调用方式:

  [A,B,C,...] =textread(filename,format)

  [A,B,C,...] =textread(filename,format,N，’headerlines’,M)

  ​	Filename：文件名；

  ​	Format：就是要读取的格式；

  ​	A,B,C…：从文件中读取到的数据。中括号里面变量的个数必须和format中定义的个数相同。

  ​	N:表示读取的次数，每次读取一行。

  ​	Headerlines：表示从第M+1行开始读入。

###5. dlmread读取方式

* a． 基本说明:

  只能读取数值数据。如文件中含有文本，需使用range参数跳过。返回值为矩阵。

* b． 调用方式:

  result =dlmread(filename,delimiter,range);

  filename：文件名。

  delimiter：分隔符。

  range：文件读取范围，格式为[R1 C1 R2 C2]。

* c． 有无分隔符限制:

  可自定义分隔符，不是必须的。

* d． 能否自定义读取范围:

  通过设置range，选择读取范围。

* e． 适用文件类型

  txt、csv。

###6. xlsread读取方式

* a． 基本说明:

  读取xls文件,可读取含文本的数据，仅能返回数值部分。

* b． 调用方式:

  [num,txt,raw]=xlsread(file,sheet,range)；

  file：需要读取的文件。

  sheet：需要读取的表单。

  range：读取范围，格式为’A1:C4’。

  num：返回的数值数据。

  txt:返回的文本数据。

  raw：返回未处理的数据。

* c． 有无分隔符限制

  无需输入分隔符。

* d． 能否自定义读取范围:

  由sheet和range定义读取范围。

* e． 文件适用范围

  xls、xlsx。

###7. csvread读取方式

* a． 基本说明:

  只能读取逗号分隔的数值数据。如文件中含有文本，需使用range参数跳过。

* b．调用方式:

  m = csvread('filename',r,c,rng)；

  filename：文件名字。

  r，c：开始读取的位置

  rng：读取范围，格式为[R1 C1 R2 C2]

* c．有无分隔符限制:

  文件必须以逗号分隔。

* d． 能否自定义读取范围:

  可由r，c ,rng定义读取范围。

* e． 文件适用范围

  txt、csv。

###8. fread读取方式

该函数读取文件返回的是二进制矩阵。

## matlab写文件函数总结

##1. xlswrite读取方式

* a.基本说明:

​     用matlab处理数据之后，需要将其保存到EXCEL内，而这必须用到xlswrite函数。

* b.调用方式

  **A=xlswrite(filename, M);** 将矩阵M的数据写入名为filename的Excel文件中。

  **B=xlswrite(filename, M, sheet)** ；将矩阵M的数据写入文件名为filename中的指定的sheet中。

  **C=xlswrite(filename, M, range)**；将矩阵M中的数据写入文件名为filename的Excel文件中，且由range制定存储的区域，例如'C1:C2'.

  **D=xlswrite(filename, M, sheet, range)**

  **status = xlswrite(filename, ...);**返回完成状态值。

  如果写入成功，则status为1;反之,写入失败，则status为0.

##2. dlmwrite读取方式

* a.基本说明:

  将矩阵写入ASCII分隔的文件。

* b.调用方式

  1）dlmwrite(filename, M)

  使用默认的分隔符(')将矩阵M写入ASCII格式的文件中。在目标文件filname中，数据是从第一行的第一列开始写的。输入的filename是使用单引号括起来的字符串。

  2）dlmwrite(filename, M, 'D')

  将矩阵M写入一个ASCII格式的文件中，使用分隔符D来分割矩阵的元素。在目标文件filname中，数据是从第一行的第一列开始写的。逗号'是默认的分隔符，使用\t来产生制表符分割的文件。

  3）dlmwrite(filename, M, 'D', R, C)

  将矩阵M写入一个ASCII格式的文件中，使用分隔符D来分割矩阵的元素。在目标文件filname中，数据是从第R行的第C列开始写的，R和C从0开始，因此R=0，C=0指定了文件中的第一个数值，即左上角的位置。

  4）dlmwrite(filename, M, '-append') 

  将矩阵数据追加到文件的末尾。如果你不指定''-append'，dlmwrite覆盖文件中的任何现有数据。
  5）dlmwrite(filename,M, '-append', attribute-value list) 
  接受一个属性值对列表。用户可以将'-append'标志放在属性-数值对之间，但不能放在属性和它的值的中间。



参考资料:

[dlmwrite的用法](https://ww2.mathworks.cn/help/matlab/ref/dlmwrite.html)

[如何使用dlmwrite将matlab中的数据保存到.txt文件中](https://blog.csdn.net/qq_41759516/article/details/82240538)

注:参考网上资料,侵权请联系删除!
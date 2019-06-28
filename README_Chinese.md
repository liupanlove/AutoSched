### 数据来源
1. 基因表达数据  
    基因表达数据链接[点击](https://1drv.ms/u/s!AkIbjcoX6iLQgTp7bSegkbYDEbBP?e=bewd1M)，其中包括GSE2109，GSE13159，GSE15061等

2. 遥感数据  
    遥感数据来源于DeepGlobe 2018。链接[点击](https://competitions.codalab.org/competitions/18468)

### 如何运行
程序主要分为两个部分，一个是数据的预处理，另一个是聚类主程序。
#### 预处理 
利用命令 ```make preprocess_data ```可以生成 preprocess_data 预处理程序  
预处理程序主要有三个参数：  
1. 第一个参数是预处理的输入数据，数据格式为每一行对应数据集的一个样本，数据间使用空格分隔开，其中每一行的第一个数据为样本点的标识（为无用数据）。  
2. 第二个参数是预处理的输出数据的路径和文件名。
3. 第三个参数为数据的维度，包括样本点的标识。

比如说下面的例子，是一个数据维度为3的3D_spatial_network.txt的数据集的预处理命令
```
bsub -b -I -q q_sw_yfb -n 1 -np 1 -cgsp 64  -host_stack 2048 -share_size 4096  ./preprocess_data  ./data/3D_spatial_network.txt ./data/3D_spatial_network.dat 4 
```
预处理最终得到的结果就是数据集的二进制表示，其中不同维度或者不同样本点之间没有任何分隔符。

#### 聚类主程序
利用命令 ```make kmeansTest ```可以生成 kmeansTest 聚类主程序。
聚类主程序主要有八个参数：
1. 第一个参数为预处理之后得到的数据；
2. 第二个参数为数据集的大小，即n；
3. 第三个参数为聚类中心的数量，即k；
4. 第四个参数为数据维度，即d；
5. 第五个参数为是否保存每一次迭代的结果，其中0表示不保存，1表示保存；
6. 第六个参数为距离文件的路径和文件名；
7. 第七个参数为是否计算评估函数，其中0表示不计算，1表示计算；
8. 第八个参数表示是否是遥感数据，其中0表示不是，1表示是；

比如说下面的例子，表示对n=4251，k=60，d=54675的数据集进行聚类，并且计算评估函数的值，其中距离文件为./data/distance.dat。
```
bsub -b -I -q q_sw_expr -n 4 -cgsp 64 -share_size 6144 -host_stack 128 ./kmeansTest  ./data/4251.dat 4251 60 54675 0 ./data/distance.dat 2 0
```

聚类主程序最终得到的结果保存在./data/round_%d_cluster.dat中，包括聚类最终得到的中心点的坐标以及每个类中的样本的数量以及可能的评估函数的值，其中%d表示k的大小。

#### 计算距离文件
由于评估函数需要样本之间的距离，对于同一个数据集不同的k的距离是一样的。我们的想法就是预先计算出样本点之间的距离存放在文件之中。  
利用命令 ```make deal_data ```可以生成 deal_data 计算距离文件的程序。  
该程序主要有三个参数：
1. 第一个参数为预处理得到的数据；
2. 第二个参数为数据集的大小，即n；
3. 第三个参数为数据维度，即d；
4. 第四个参数为输出的距离文件的路径和文件名；
比如说下面的例子：
```
bsub -b -I -q q_sw_expr -host_stack 1024 -N 2 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000  ./deal_data ./data/4251.dat 4251 54675 ./data/distance.dat
```


### 组织结构
data 文件夹主要用于存放输入输出的数据  
util 文件夹为基于神威太湖之光的底层优化程序
preprocess_data.c 为预处理程序  
deal_data.c deal_data_slave.c 分别为计算距离文件的主核程序和从核程序  
master.c, slave.c 分别为聚类主程序的主核程序和从核程序  
*.sh为一些运行辅助脚本
该目录下的代码用于为数据集添加词性(POS)特征和依存分析(DP)特征

使用方式：

分割数据集（9-1分）： 

python additional_features.py -sp -rf <path/to/your/data> -train <save_path/of/training_set> -test <save_path/of/testing_set> -ltp <path/to/ltp_Data_v3.4.0>

不分割数据集：

python additional_features.py -rf <path/to/your/data> -wf <save_path/of/data> -ltp <path/to/ltp_Data_v3.4.0>




These codes are used to add the POS and dependency parsing (DP) features into the data set

Usage：

split the data set to training and testing set (0.9-0.1)： 

python additional_features.py -sp -rf <path/to/your/data> -train <save_path/of/training_set> -test <save_path/of/testing_set> -ltp <path/to/ltp_Data_v3.4.0>

do not split the data set：

python additional_features.py -rf <path/to/your/data> -wf <save_path/of/data> -ltp <path/to/ltp_Data_v3.4.0>

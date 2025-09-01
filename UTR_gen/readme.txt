---------------------
data文件夹里是数据集：按照train、val和test分为三个子文件夹，子文件夹中包含cds、5utr和3utr序列。
---------------------
bert-base-cased文件夹包含了对应的预训练模型的参数和权重文件。
---------------------
bert-base-cased文件夹中的vocab_5utr.txt和vocab_3utr.txt是根据K-mers特征构建的字典，适用于所有模型的tokenize过程。
---------------------
result文件夹中包含训练得到的模型结果和评价结果。目录下保存的是位置编码为2048的bert-base-cased模型结果；bert-base-cased子文件夹中的512文件夹包含了位置编码为512的bert-base-cased模型结果；RNA-FM、CodonBERT和cdsBERT三个子文件夹包含的是对应模型的结果。
最优模型是位置编码为2048的bert-base-cased模型结果中的cds25utr_model.98.bin和cds23utr_model.93.bin。
---------------------
cds25utr_train.py、cds25utr_test.py、cds23utr_train.py和cds23utr_test.py这四个文件分别对应5utr和3utr的训练和测试代码。
cds25utr_gen.py和cds23utr_gen.py为对应的UTR生成代码，用于实际的UTR设计任务。
token.py代码可使用选定的模型结果对序列文件进行单独tokenize操作，得到片段化的序列文本，分析时用于片段化原始数据文件，以匹配生成的文件格式。
运行对应代码的示例：CUDA_VISIBLE_DEVICES=0  python cds25utr_train.py 
---------------------
preprocess.ipynb包含了数据处理流程、K-mers词典构建流程、Rouge评价、Jaccard评价、BLEU评价和K-mers分析。
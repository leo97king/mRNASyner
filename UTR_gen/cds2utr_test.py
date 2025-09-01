import torch
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
import os
import numpy as np
import random
def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

seed_torch(12345)
def load_vocab(vocab_path, simplfied=False, startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]):
    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    word2ix = {}
    for index, line in enumerate(lines):
        word2ix[line.strip("\n")] = index
    return word2ix
##100epoch最佳模型为cds25utr_model.98.bin和cds23utr_model.93.bin
cds2utr_model = "/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/result/cds23utr_model.98.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#test_file=open('/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/data/test/test_cds.txt')
if __name__ == "__main__":
    vocab_path = "/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/bert-base-cased/vocab_5utr.txt"  # bert模型字典的位置
    model_name = "bert"  # 选择模型名字
    # 加载字典
    word2ix = load_vocab(vocab_path)
    # 定义模型
    bert_model = load_bert(word2ix, model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()
    ## 加载训练的模型参数～
    bert_model.load_all_params(model_path=cds2utr_model, device=device)
    #print(bert_model.word2ix)
    #print(bert_model.config.max_position_embeddings)
    test_file=open('/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/data/test/test_cds_unix.txt','r')
    test_data = test_file.readlines()
    #test_data=["ATGGCCTTCTCAGGCCGAGCGCGCCCCTGCATTATCCCAGAGAACGAAGAAATCCCCCGAGCAGCCCTTAACACTGTCCACGAGGCCAATGGGACCGAGGACGAGAGGGCTGTTTCCAAACTGCAGCGCAGGCACAGTGACGTGAAAGTCTACAAGGAGTTCTGTGACTTTTATGCGAAATTCAACATGGCCAACGCCCTGGCCAGCGCCACTTGCGAGCGCTGCAAGGGCGGCTTTGCGCCCGCTGAGACGATCGTGAACAGTAATGGGGAGCTGTACCATGAGCAGTGTTTCGTGTGCGCTCAGTGCTTCCAGCAGTTCCCAGAAGGACTCTTCTATGAGGAACGAACGTGA",
               #"ATGAAGGGGAGCCGTGCCCTCCTGCTGGTGGCCCTCACCCTGTTCTGCATCTGCCGGATGGCCACAGGGGAGGACAACGATGAGTTTTTCATGGACTTCCTGCAAACACTACTGGTGGGGACCCCAGAGGAGCTCTATGAGGGGACCTTGGGCAAGTACAATGTCAACGAAGATGCCAAGGCAGCAATGACTGAACTCAAGTCCTGTATAGATGGCCTGCAGCCAATGCACAAGGCGGAGCTGGTCAAGCTGCTGGTGCAAGTGCTGGGCAGTCAGGACGGTGCCTAA",
               #"ATGATTAGCTCAGTAAAACTCAATCTCATCCTAGTTCTGTCGCTGTCCACAATGCATGTGTTTTGGTGTTATCCAGTTCCATCTTCTAAGGTGTCTGGAAAATCTGATTACTTTCTCATTCTGCTGAACAGCTGCCCAACCAGATTGGACAGGAGCAAAGAACTAGCTTTTCTAAAGCCAATTTTGGAGAAGATGTTTGTGAAAAGGTCCTTTCGCAATGGAGTTGGCACAGGGATGAAAAAAACTTCCTTTCAAAGAGCAAAATCATGA"]
    with open('/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/result/bert_2048_3utr.txt','w') as f1:
        for text in test_data:
            with torch.no_grad():
                #print(word2ix["[SEP]"])
                # for k in range(1,50):
                f1.write(bert_model.generate(text,beam_size=10))
                f1.write('\n')
    f1.close()           
                
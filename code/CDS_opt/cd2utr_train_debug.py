import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

#from transformers import BertPreTrainedModel, BertModel
import os
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import torch.nn as nn

def load_vocab(vocab_path, simplfied=False, startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]):
    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    word2idx = {}
    for index, line in enumerate(lines):
        word2idx[line.strip("\n")] = index
    return word2idx

vocab_path = "D:/bert_seq2seq/bert-base-cased/vocab_5utr.txt"  # bert模型字典的位置
word2idx = load_vocab(vocab_path, simplfied=False)
# tokenizer = BertTokenizer.from_pretrained("D:/bert_seq2seq/bert-base-cased/vocab.txt")
# word2idx = tokenizer.get_vocab()
model_name = "bert"  # 选择模型名字
model_path = "D:/bert_seq2seq/bert-base-cased/pytorch_model.bin"  # 模型位置
model_save_dir = "D:/bert_seq2seq/result/"
batch_size = 4
lr = 1e-5
maxlen=2048


def read_file(src_dir, tgt_dir):
    src = []
    tgt = []

    with open(src_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            src.append(line.strip('\n'))

    with open(tgt_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n'))

    return src,tgt

def rouge_score(pre_list,true_list):
    rouge = Rouge()
    for i in range(len(pre_list)):
        if pre_list[i]=='' or pre_list==' ':
            pre_list[i]='#'
    rouge_scores = rouge.get_scores(pre_list, true_list,avg=True)
    return rouge_scores

def jaccard_similarity(set1,set2):
    # 计算两个集合的交集
    intersection = len(set1.intersection(set2))
    # 计算两个集合的并集
    union = len(set1.union(set2))
    # 计算Jaccard系数
    jaccard_coefficient = intersection / union
    return jaccard_coefficient
def jaccard_set(seq):
    return(set(seq.split()))
def jaccard_score(pre_list,true_list):
    jaccard_list=[]
    for i in range(len(pre_list)):
        jaccard_coefficient = jaccard_similarity(jaccard_set(pre_list[i]), jaccard_set(true_list[i]))
        jaccard_list.append(jaccard_coefficient)
    jaccard_scores=sum(jaccard_list)/len(jaccard_list)
    return jaccard_scores

def bleu_score(pre_list,true_list):
    candidates=[]
    references=[]
    for i in range(len(pre_list)):
        if pre_list[i]=='' or pre_list==' ':
            pre_list[i]='#'
    for x in pre_list:
        candidates.append(list(x.split(' ')))
    for y in true_list:        
        references.append(list(y.split(' ')))
    return corpus_bleu(references,candidates) 

class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt, max_length=maxlen)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)
        
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

class Trainer:
    def __init__(self):
        # 加载数据
        src_dir = 'D:/bert_seq2seq/data/train/train_cds.txt'
        tgt_dir = 'D:/bert_seq2seq/data/train/train_5utr.txt'
        src_eval_dir='D:/bert_seq2seq/data/val/val_cds.txt'
        tgt_eval_dir='D:/bert_seq2seq/data/val/val_5utr.txt'
        self.sents_src, self.sents_tgt= read_file(src_dir, tgt_dir)
        self.sents_eval_src,self.sents_eval_tgt=read_file(src_eval_dir,tgt_eval_dir)
        # self.sents_src= torch.load("./corpus/auto_title/train_clean.src")
        # self.sents_tgt = torch.load("./corpus/auto_title/train_clean.tgt")

        # 判断是否有可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="seq2seq")
        
        self.bert_model.set_device(self.device)
        ## 加载预训练的模型参数～  
        self.bert_model.load_pretrain_params(model_path)
        self.add_position_embeddings = torch.tensor(self.bert_model.bert.embeddings.new_position_embeddings.weight)
        self.add_position_embeddings[:512] = torch.tensor(self.bert_model.bert.embeddings.position_embeddings.weight)
        self.add_position_embeddings[512:1024] = torch.tensor(self.bert_model.bert.embeddings.position_embeddings.weight)
        self.add_position_embeddings[1024:1536] = torch.tensor(self.bert_model.bert.embeddings.position_embeddings.weight)
        self.add_position_embeddings[1536:2048] = torch.tensor(self.bert_model.bert.embeddings.position_embeddings.weight)
        self.bert_model.bert.embeddings.new_position_embeddings.weight = nn.Parameter(self.add_position_embeddings)
        print(self.bert_model.bert.embeddings.position_embeddings.weight,self.bert_model.bert.embeddings.new_position_embeddings.weight)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = BertDataset(self.sents_src, self.sents_tgt)
        dataset_eval=BertDataset(self.sents_eval_src, self.sents_eval_tgt)
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.dataloader_eval =  DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        #print(Tokenizer(word2idx).tokenize('AUGAAGCCAGUAAGUCGUCGCACGCUGGACUGGAUUUAUUCAGUGUUGCUGCUUGCCAUCGUUUUAAUCUCCUGGGGCUGCAUCAUCUAUGCUUCGAUGGUGUCUGCAAGACGACAGCUAAGGAAGAAAUACCCAGACAAAAUCUUUGGGACGAAUGAAAAUUUGUAA'))

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        tr_loss=self.iteration(epoch, dataloader=self.dataloader, train=True)
        
        return tr_loss
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))
    
    def eval_step(self,epoch):
        predicts=[]
        trues=[]
        val_loss = 0
        dataloader_eval=self.dataloader_eval
        test_data = self.sents_eval_src
        true_data=self.sents_eval_tgt
        self.bert_model.eval()
        tokenizer=Tokenizer(word2idx)
        s=0
        with torch.no_grad():
            if s<=3:
                for token_ids, token_type_ids, target_ids in dataloader_eval:
                    predictions, loss = self.bert_model(token_ids,
                                                            token_type_ids,
                                                            labels=target_ids,
                                                            )
                    val_loss+=loss.item()
                    s=s+1
                print("epoch is " + str(epoch)+"val_loss is " + str(val_loss))
                #生成两个token级别的序列pre_list和true_list
            for text in test_data[:2]:
                generater=trainer.bert_model.generate(text, beam_size=5)
                predicts.append(generater)
            for text_true in true_data[:2]:
                true_list=tokenizer.tokenize(text_true)
                true_list=true_list[1:-1]
                s=''
                n=1
                for data in true_list:
                    if n==1:
                        s=s+data
                    else:
                        s=s+' '+data
                    n+=1
                trues.append(s)
            #计算验证集的指标评价
            try:
                rouge_scores=rouge_score(predicts,trues)
            except:
                rouge_scores=0
            try:
                jaccard_scores=jaccard_score(predicts,trues)
            except:
                jaccard_scores=0
            try:
                bleu_scores=bleu_score(predicts,trues)
            except:
                bleu_scores=0
            print("rouge_score:{},jaccard_score:{},blue_score:{}".format(rouge_scores,jaccard_scores,bleu_scores))
        return val_loss,jaccard_scores,rouge_scores['rouge-1'],bleu_scores

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 800 == 0:
                self.bert_model.eval()
                test_data = ["AUGAAGCCAGUAAGUCGUCGCACGCUGGACUGGAUUUAUUCAGUGUUGCUGCUUGCCAUCGUUUUAAUCUCCUGGGGCUGCAUCAUCUAUGCUUCGAUGGUGUCUGCAAGACGACAGCUAAGGAAGAAAUACCCAGACAAAAUCUUUGGGACGAAUGAAAAUUUGUAA",
                 "AUGAAUUGGAAGGUUCUUGAGCACGUGCCCCUGCUGCUGUAUAUCUUGGCAGCAAAAACAUUAAUUCUCUGCCUGACAUUUGCUGGGGUGAAAAUGUAUCAAAGAAAAAGGUUGGAGGCAAAACAACAAAAACUGGAGGCUGAAAGGAAGAAGCAAUCAGAGAAAAAAGAUAACUGA"]
                for text in test_data:
                    print(self.bert_model.generate(text, beam_size=3))
                print("loss is " + str(report_loss))
                report_loss = 0
                self.bert_model.train()


            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                )
            report_loss += loss.item()
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch)+". loss is " + str(total_loss) + ". spend time is "+ str(spend_time))
        # 保存模型
        return total_loss

if __name__ == '__main__':

    trainer=Trainer()
    
    # train_epoches = 100
    # tr_loss_list,val_loss_list,jaccard_list,rouge_list,bleu_list=[],[],[],[],[]
    # best_score=0
    # best_epoch=1
    # for epoch in range(100):
    #     # 训练一个epoch
    #     #tr_loss=trainer.train(epoch)
    #     val_loss,val_jaccard,val_rouge,val_bleu=trainer.eval_step(epoch)
    #     val_scores=(val_jaccard + val_bleu + (val_rouge['r']+val_rouge['p']+val_rouge['f'])/3)/3
    #     print(val_scores)
    #     tr_loss_list.append(tr_loss)
    #     val_loss_list.append(val_loss)
    #     jaccard_list.append(val_jaccard)
    #     rouge_list.append(val_rouge)
    #     bleu_list.append(val_bleu)
    #     if  val_scores > best_score:
    #         best_score,best_epoch=val_scores,epoch
    #         model_save_path=os.path.join(model_save_dir,"cds25utr_model.{}.bin".format(epoch))
    #         trainer.save(model_save_path)
    #     tr_pd=pd.DataFrame(tr_loss_list)
    #     val_pd=pd.DataFrame(val_loss_list)
    #     jaccard_pd=pd.DataFrame(jaccard_list)
    #     rouge_pd=pd.DataFrame(rouge_list)
    #     bleu_pd=pd.DataFrame(bleu_list)
    #     tr_pd.to_csv('/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/result/tr_loss_5utr.csv')
    #     val_pd.to_csv('/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/result/val_loss_5utr.csv')
    #     jaccard_pd.to_csv('/data2/zyfeng/zjgu/unilm/unilm/bert_seq2seq/result/jaccard_5utr.csv')
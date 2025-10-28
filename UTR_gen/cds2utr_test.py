import torch
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
import os
import numpy as np
import random
import argparse

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--in_dir', type=str, default='./data/', help='Path to the input TXT data file.')
    parser.add_argument('--out_dir', type=str, default='./result/', help='Save the output file.')

    args = parser.parse_args()
    test_file=open(args.in_dir+'opt_cds.txt','r')
    test_data = test_file.readlines()
    
    #3UTR
    cds2utr_model = "./model/cds23utr_model.bin"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = "./bert-base-cased/vocab_3utr.txt"  
    model_name = "bert"  

    word2ix = load_vocab(vocab_path)

    bert_model = load_bert(word2ix, model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()

    bert_model.load_all_params(model_path=cds2utr_model, device=device)
    print('Generating 3UTR...')
    
    with open('./result/3utr.txt','w') as f1:
        for text in test_data:
            with torch.no_grad():
                f1.write(bert_model.generate(text,beam_size=10).replace(' ','').replace('T','U')+'\n')
    f1.close()   
    del bert_model
    #5UTR
    cds2utr_model = "./model/cds25utr_model.bin"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = "./bert-base-cased/vocab_5utr.txt"  
    model_name = "bert"  

    word2ix = load_vocab(vocab_path)

    bert_model = load_bert(word2ix, model_name=model_name)
    bert_model.set_device(device)
    bert_model.eval()

    bert_model.load_all_params(model_path=cds2utr_model, device=device)
    print('Generating 5UTR...')
    
    with open('./result/5utr.txt','w') as f2:
        for text in test_data:
            with torch.no_grad():

                f2.write(bert_model.generate(text,beam_size=10).replace(' ','').replace('T','U')+'\n')
    f2.close()        
    print(f'Done!Results are saved in /UTR_gen/result/3utr.txt and /UTR_gen/result/5utr.txt')
    del bert_model
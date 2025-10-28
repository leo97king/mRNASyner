
import torch
import torch.nn as nn
from bert_seq2seq.tokenizer import Tokenizer
    
def get_model(model_name, word2ix):
    if model_name == "roberta":
        from bert_seq2seq.model.roberta_model import BertModel, BertConfig, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead, CLS
        config = BertConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)

        CLS = CLS(config)

    elif model_name == "bert":
        from bert_seq2seq.model.bert_model import BertConfig, BertModel, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead,CLS
        config = BertConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)

    elif model_name == "nezha":
        from bert_seq2seq.model.nezha_model import BertConfig, BertModel, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead,CLS
        config = BertConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)

    elif model_name == "roberta-large":
        from bert_seq2seq.model.roberta_model import BertModel, RobertaLargeConfig, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead, CLS
        config = RobertaLargeConfig(vocab_size=len(word2ix))
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)
        
    else:
        raise Exception("model_name_err")

    return config, bert, layer_norm_cond, CLS

class BasicBert(nn.Module):
    def __init__(self, word2ix, model_name="roberta", tokenizer=None):
        super().__init__()
        self.config = ""
        self.word2ix = word2ix

        if tokenizer is None:
            self.tokenizer = Tokenizer(word2ix)
        else:
            self.tokenizer = tokenizer

        self.model_name = model_name
        
        self.config, self.bert, self.layer_norm_cond, self.cls = get_model(model_name, word2ix)
       
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path, keep_tokens=None, strict=False):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)

        checkpoint = {k: v for k, v in checkpoint.items()
                                            }
        #删除预训练模型中的词典向量
        # print(checkpoint['criterion'])
        # print(checkpoint['task_state'].keys())
        # print(checkpoint['model']['encoder.encoder.embed_tokens.weight'].size())
        # print(checkpoint['model']['encoder.encoder.lm_head.weight'].size())
        # print(checkpoint['model']['encoder.encoder.lm_head.bias'].size())
        # print(checkpoint['model']['encoder.encoder.lm_head.dense.weight'].size())
        # print(checkpoint['model']['encoder.encoder.lm_head.dense.bias'].size())
        # print(checkpoint['model']['encoder.encoder.lm_head.layer_norm.weight'].size())
        # print(checkpoint['model']['encoder.encoder.lm_head.layer_norm.bias'].size())
        # print(checkpoint['encoder.encoder.embed_positions.weight'])
        # print(checkpoint['model']['encoder.encoder.emb_layer_norm_after.weight'].size())
        # print(checkpoint['model']['encoder.encoder.emb_layer_norm_after.bias'].size())


        #FM_keys=['args', 'cfg', 'model', 'criterion', 'optimizer_history', 'task_state', 'extra_state', 'last_optimizer_state']
        #del_list=[cls.predictions.bias,cls.predictions.decoder.weight,cls.predictions.decoder.bias]
        # print(checkpoint['cls.predictions.bias'])
        # print(checkpoint['cls.predictions.decoder.weight'])
        # print(checkpoint['cls.predictions.decoder.bias'])
        # print(checkpoint['bert.embeddings.word_embeddings.weight'])        
        # sub1=checkpoint.pop('cls.predictions.bias')
        # sub2=checkpoint.pop('cls.predictions.decoder.weight')
        # sub3=checkpoint.pop('cls.predictions.decoder.bias')
        # sub4=checkpoint.pop('bert.embeddings.word_embeddings.weight')
        # CondonBERT初始化(参数名称相同，但维度不同)
        # sub1_1=sub1.repeat(28996//130)
        # sub1_2=sub1.narrow(0,0,28996%130)
        # checkpoint['cls.predictions.bias']=torch.cat((sub1_1,sub1_2),0)
        # sub2_1=sub2.repeat(28996//130,1)
        # sub2_2=sub2.narrow(0,0,28996%130)
        # checkpoint['cls.predictions.decoder.weight']=torch.cat((sub2_1,sub2_2),0)
        # sub3_1=sub3.repeat(28996//130)
        # sub3_2=sub3.narrow(0,0,28996%130)
        # checkpoint['cls.predictions.decoder.bias']=torch.cat((sub3_1,sub3_2),0)
        # sub4_1=sub4.repeat(28996//130,1)
        # sub4_2=sub4.narrow(0,0,28996%130)
        # checkpoint['bert.embeddings.word_embeddings.weight']=torch.cat((sub4_1,sub4_2),0)   

        ##将预训练模型与框架中命名不一致的参数匹配
        #CodonBERT
        # checkpoint['bert.embeddings.LayerNorm.gamma']=checkpoint['bert.embeddings.LayerNorm.weight']
        # checkpoint['bert.embeddings.LayerNorm.beta']=checkpoint['bert.embeddings.LayerNorm.bias']
        # checkpoint['cls.predictions.transform.LayerNorm.gamma']=checkpoint['cls.predictions.transform.LayerNorm.weight']
        # checkpoint['cls.predictions.transform.LayerNorm.beta']=checkpoint['cls.predictions.transform.LayerNorm.bias']
        # for i in range(12):
        #     layer1_name=f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        #     layer1_pre_name=f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight'
        #     layer2_name=f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        #     layer2_pre_name=f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias'
        #     checkpoint[layer1_name]=checkpoint[layer1_pre_name]
        #     checkpoint[layer2_name]=checkpoint[layer2_pre_name]
        # print(checkpoint['bert.encoder.layer.0.attention.output.LayerNorm.gamma'])
        # print(checkpoint['bert.encoder.layer.0.attention.output.LayerNorm.beta'])
        # print(checkpoint['bert.embeddings.LayerNorm.gamma'])
        # print(checkpoint['bert.embeddings.LayerNorm.beta'])
        # print(checkpoint['cls.predictions.transform.LayerNorm.gamma'])
        # print(checkpoint['cls.predictions.transform.LayerNorm.beta'])

        #RNA-FM初始化(参数名称相同，但维度不同)
        # sub1=checkpoint.pop('encoder.encoder.embed_tokens.weight')
        # sub2=checkpoint.pop('encoder.encoder.lm_head.weight')
        # sub3=checkpoint.pop('encoder.encoder.lm_head.bias')
        # sub4=checkpoint.pop('encoder.encoder.embed_positions.weight')
        # sub1_1=sub1.repeat(28996//25,1)
        # sub1_2=sub1.narrow(0,0,28996%25)
        # checkpoint['bert.embeddings.word_embeddings.weight']=torch.cat((sub1_1,sub1_2),0)
        # sub2_1=sub2.repeat(28996//25,1)
        # sub2_2=sub2.narrow(0,0,28996%25)
        # checkpoint['cls.predictions.decoder.weight']=torch.cat((sub2_1,sub2_2),0)
        # sub3_1=sub3.repeat(28996//25)
        # sub3_2=sub3.narrow(0,0,28996%25)
        # checkpoint['cls.predictions.bias']=torch.cat((sub3_1,sub3_2),0)
        # checkpoint['cls.predictions.decoder.bias']=torch.cat((sub3_1,sub3_2),0)
        # checkpoint['bert.embeddings.position_embeddings.weight']=sub4.narrow(0,2,1024)
                                      
        #RNA-FM
        # checkpoint['cls.predictions.transform.dense.weight']=checkpoint['encoder.encoder.lm_head.dense.weight']
        # checkpoint['cls.predictions.transform.dense.bias']=checkpoint['encoder.encoder.lm_head.dense.bias']
        # checkpoint['cls.predictions.transform.LayerNorm.gamma']=checkpoint['encoder.encoder.lm_head.layer_norm.weight']
        # checkpoint['cls.predictions.transform.LayerNorm.beta']=checkpoint['encoder.encoder.lm_head.layer_norm.bias']
        # checkpoint['bert.embeddings.LayerNorm.gamma']=checkpoint[ 'encoder.encoder.emb_layer_norm_after.weight']
        # checkpoint['bert.embeddings.LayerNorm.beta']=checkpoint['encoder.encoder.emb_layer_norm_after.bias']
        # for i in range(12):
        #     fm1_name=f'bert.encoder.layer.{i}.attention.self.query'
        #     fm1_pre_name=f'encoder.encoder.layers.{i}.self_attn.q_proj'
        #     fm2_name=f'bert.encoder.layer.{i}.attention.self.key'
        #     fm2_pre_name=f'encoder.encoder.layers.{i}.self_attn.k_proj'
        #     fm3_name=f'bert.encoder.layer.{i}.attention.self.value'
        #     fm3_pre_name=f'encoder.encoder.layers.{i}.self_attn.v_proj'
        #     fm4_name=f'bert.encoder.layer.{i}.attention.output.dense'
        #     fm4_pre_name=f'encoder.encoder.layers.{i}.self_attn.out_proj'
        #     fm5_name=f'bert.encoder.layer.{i}.attention.output.LayerNorm'
        #     fm5_pre_name=f'encoder.encoder.layers.{i}.self_attn_layer_norm'
        #     fm6_name=f'bert.encoder.layer.{i}.intermediate.dense'
        #     fm6_pre_name=f'encoder.encoder.layers.{i}.fc1'
        #     fm7_name=f'bert.encoder.layer.{i}.output.dense'
        #     fm7_pre_name=f'encoder.encoder.layers.{i}.fc2'
        #     fm8_name=f'bert.encoder.layer.{i}.output.LayerNorm'
        #     fm8_pre_name=f'encoder.encoder.layers.{i}.final_layer_norm'                                                                         
        #     checkpoint[fm1_name+'.weight']=checkpoint[fm1_pre_name+'.weight']
        #     checkpoint[fm2_name+'.weight']=checkpoint[fm2_pre_name+'.weight']
        #     checkpoint[fm3_name+'.weight']=checkpoint[fm3_pre_name+'.weight']
        #     checkpoint[fm4_name+'.weight']=checkpoint[fm4_pre_name+'.weight']
        #     checkpoint[fm5_name+'.gamma']=checkpoint[fm5_pre_name+'.weight']
        #     checkpoint[fm6_name+'.weight']=checkpoint[fm6_pre_name+'.weight']
        #     checkpoint[fm7_name+'.weight']=checkpoint[fm7_pre_name+'.weight']
        #     checkpoint[fm8_name+'.gamma']=checkpoint[fm8_pre_name+'.weight']
        #     checkpoint[fm1_name+'.bias']=checkpoint[fm1_pre_name+'.bias']
        #     checkpoint[fm2_name+'.bias']=checkpoint[fm2_pre_name+'.bias']
        #     checkpoint[fm3_name+'.bias']=checkpoint[fm3_pre_name+'.bias']
        #     checkpoint[fm4_name+'.bias']=checkpoint[fm4_pre_name+'.bias']
        #     checkpoint[fm5_name+'.beta']=checkpoint[fm5_pre_name+'.bias']
        #     checkpoint[fm6_name+'.bias']=checkpoint[fm6_pre_name+'.bias']
        #     checkpoint[fm7_name+'.bias']=checkpoint[fm7_pre_name+'.bias']
        #     checkpoint[fm8_name+'.beta']=checkpoint[fm8_pre_name+'.bias']         
        
        if keep_tokens is not None:
            ## 说明精简词表了，embeedding层也要过滤下
            embedding_weight_name = "bert.embeddings.word_embeddings.weight"
            cls_pre_weight = "cls.predictions.decoder.weight"
            cls_pre_bias = "cls.predictions.bias"
            checkpoint[embedding_weight_name] = checkpoint[embedding_weight_name][keep_tokens]
            checkpoint[cls_pre_weight] = checkpoint[cls_pre_weight][keep_tokens]
            checkpoint[cls_pre_bias] = checkpoint[cls_pre_bias][keep_tokens]
            
        self.load_state_dict(checkpoint, strict=strict)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):

        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, input_text):
        ## 返回bert编码后得到的向量
        input_ids, _ = self.tokenizer.encode(input_text, max_length=512)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).view(1, -1)

        enc_layers, _ = self.bert(input_ids, position_ids=None, token_type_ids=None, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[-1] ## 取出来最后一层输出 (batch, seq_len, 768)

        tokens_hidden_state, _ = self.cls(squence_out)

        return tokens_hidden_state

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)
        
    def save_all_params(self, save_path):
        if self.model_name == "nezha":
            # 不要保存相对位置编码权重
            checkpoints = {k: v for k, v in self.state_dict().items()
                                        if "relative" not in k}
            torch.save(checkpoints, save_path)
            return
        torch.save(self.state_dict(), save_path)

class BasicGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)
        
    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)


class BasicT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

class BasicBart(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)


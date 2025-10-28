import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Network import *
import yaml
from tqdm import tqdm

def merge_cds_utr(cds_path, utr_path, output_path):
    with open(f"{cds_path}opt_cds.txt", 'r') as cds_file, \
         open(f"{utr_path}3utr.txt", 'r') as utr3_file, \
         open(f"{utr_path}5utr.txt", 'r') as utr5_file:
        

        cds_lines = [line.strip() for line in cds_file if line.strip()]
        utr3_lines = [line.strip() for line in utr3_file if line.strip()]
        utr5_lines = [line.strip() for line in utr5_file if line.strip()]

    if not (len(cds_lines) == len(utr3_lines) == len(utr5_lines)):
        raise ValueError("Number of lines in CDS, 3'UTR, and 5'UTR files do not match!")
    
    merged_data = []
    for i in range(len(cds_lines)):
        merged_seq = utr5_lines[i] + cds_lines[i] + utr3_lines[i]
        merged_data.append({
            "id": str(i + 1), 
            "sequence": merged_seq
        })
    
    df = pd.DataFrame(merged_data)
    df.to_csv(output_path+"result.csv", index=False)



class RNA2D_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in self.data.loc[idx,'sequence']]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        return {'sequence':sequence}
    

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(** config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.decoder = nn.Linear(config.ninp,5)
        
    def forward(self,src):
        
       
        sequence_features, pairwise_features=self.get_embeddings(
            src, 
            torch.ones_like(src).long().to(src.device)  
        )
        output=self.decoder(sequence_features)

        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f" {torch.cuda.get_device_name(0)}")

    merge_cds_utr('../CDS_opt/result/','../UTR_gen/result/','../result/')

    config = load_config_from_yaml('./RibonanzaNet2D_Final/configs/pairwise.yaml')
    model = finetuned_RibonanzaNet(config)
    model.load_state_dict(torch.load(
        './RibonanzaNet_Weights/RibonanzaNet-Deg.pt',
        map_location=device 
    ))
    model = model.to(device)  

    # 4. 加载测试数据
    test_data = pd.read_csv('../result/result.csv')
    test_dataset = RNA2D_Dataset(test_data)


    test_preds = []
    print('Predicting seq deg probability...')
    for i in tqdm(range(len(test_dataset))):
        example = test_dataset[i]

        sequence = example['sequence'].unsqueeze(0).to(device)  

        with torch.no_grad(): 
            test_preds.append(model(sequence).cpu().numpy())


    preds = []
    ids = []
    for i in range(len(test_data)):
        preds.append(test_preds[i][0,:])
        id = test_data.loc[i,'id']
        ids.extend([f"{id}_{pos}" for pos in range(len(test_preds[i][0,:]))])
    
    preds = np.concatenate(preds)
    
    sub = pd.DataFrame()
    sub['id_seq_pos'] = ids
    for i,l in enumerate(['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']):
        sub[l] = preds[:,i]
    sub.to_csv('../result/deg_pred.csv',index=False)
    print('Save deg Probibilty to ../result/deg_pred.csv')
    print('Done!')
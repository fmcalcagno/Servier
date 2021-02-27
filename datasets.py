import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *
from sklearn.preprocessing import OneHotEncoder
from read_smiles import *
import networkx as nx
PAD = 0
MAX_LEN = 220


class molDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


class SmilesDatasetP1(Dataset):

    def __init__(self, hotencoder,file, seq_len=2048, input_type="Float"):
        self.smiles = pd.read_csv(file)['smiles']
        self.targets = pd.read_csv(file)['P1']
        self.one_hot_targets = hotencoder.fit_transform(self.targets.values.reshape(-1, 1)).toarray()
        self.seq_len = seq_len
        self.input_type=input_type

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        mol = read_smiles(sm) # Using features from readsmiles.py
        x = nx.to_numpy_array(mol).flatten('F')
        x = np.pad(x, (0, self.seq_len-len(x)), 'constant', constant_values=(0))
        output = self.targets[item].astype(float)
        output_one_hot = self.one_hot_targets [item].astype(float)

        if self.input_type == "Float":
            x = torch.FloatTensor(x)
        else:
            x = torch.LongTensor(x)

        return x, output_one_hot


class SmilesDatasetP1_P9(Dataset):
    def __init__(self, hotencoder,file,  seq_len=2048, input_type="Float"):
        self.smiles = pd.read_csv(file)['smiles']
        self.output1 = pd.read_csv(file)['P1']
        self.output2 = pd.read_csv(file)['P2']
        self.output3 = pd.read_csv(file)['P3']
        self.output4 = pd.read_csv(file)['P4']
        self.output5 = pd.read_csv(file)['P5']
        self.output6 = pd.read_csv(file)['P6']
        self.output7 = pd.read_csv(file)['P7']
        self.output8 = pd.read_csv(file)['P8']
        self.output9 = pd.read_csv(file)['P9']
        self.outputs = pd.concat([self.output1,self.output2,self.output3,self.output4,self.output5,self.output6,self.output7,self.output8,self.output9],axis=1)
        self.one_hot_targets = hotencoder.fit_transform(self.outputs.values).toarray()
        self.input_type= input_type
        self.seq_len = seq_len


    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        mol = read_smiles(self.smiles[item])
        x = nx.to_numpy_array(mol).flatten('F')
        x = np.pad(x, (0, self.seq_len - len(x)), 'constant', constant_values=(0))
        output_one_hot = self.one_hot_targets[item].astype(float)
        if self.input_type == "Float":
            x = torch.FloatTensor(x)
        else:
            x = torch.LongTensor(x)
        return x, torch.FloatTensor(output_one_hot)



class SmilesDataset_singleline():
    def __init__(self, input1, seq_len=2048):
        self.input1 = input1

        self.seq_len = seq_len


    def get_item(self):
        mol = read_smiles(self.input1)
        x = nx.to_numpy_array(mol).flatten('F')
        x = np.pad(x, (0, self.seq_len - len(x)), 'constant', constant_values=(0))
        x = torch.LongTensor(x)

        return x
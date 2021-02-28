from torch.utils.data import Dataset, DataLoader
from utilsfun import *
from read_smiles import *
import networkx as nx
import feature_extractor as fe
import pandas as pd
import numpy as np
import torch

class SmilesDataset(Dataset):
    """Smiles Dataset Generated for all three models"""

    def __init__(self,model_number, hotencoder,file, seq_len=2048, input_type="Float"):
        self.model_number = model_number
        self.smiles = pd.read_csv(file)['smiles']
        if self.model_number==1 or  self.model_number==2:
            self.targets = pd.read_csv(file)['P1']
            self.one_hot_targets = hotencoder.fit_transform(self.targets.values.reshape(-1, 1)).toarray()
        elif self.model_number==3:
            self.targets = pd.read_csv(file)[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']]
            self.one_hot_targets = hotencoder.fit_transform(self.targets.values).toarray()
        self.seq_len = seq_len
        self.input_type=input_type

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        if self.model_number ==1:
            features = fe.fingerprint_features(sm)
            x= np.asarray(features)
        elif self.model_number >=2:
            mol = read_smiles(sm)  # Using features from readsmiles.py
            x = nx.to_numpy_array(mol).flatten('F')
            x = np.pad(x, (0, self.seq_len - len(x)), 'constant', constant_values=(0))

        output_one_hot = self.one_hot_targets [item].astype(float)
        if self.input_type == "Float":
            x = torch.FloatTensor(x)
        else:
            x = torch.LongTensor(x)

        if self.model_number==3:
            output_one_hot = torch.FloatTensor(output_one_hot)

        return x, output_one_hot


#
class SmilesDataset_singleline():
    """Dataset Generated to predict a single molecule properties"""
    def __init__(self, model_number,input1, seq_len=2048):
        self.input1 = input1

        self.seq_len = seq_len
        self.model_number = model_number

    def get_item(self):
        if self.model_number == 1:
            features = fe.fingerprint_features(self.input1)
            x = np.asarray(features)
        elif self.model_number >= 2:
            mol = read_smiles(self.input1)  # Using features from readsmiles.py
            x = nx.to_numpy_array(mol).flatten('F')
            x = np.pad(x, (0, self.seq_len - len(x)), 'constant', constant_values=(0))

        x = torch.LongTensor(x)

        return x


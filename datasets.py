import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from enumerator import SmilesEnumerator
from utils import split

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


class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()

    def __call__(self, sm):
        sm_r = self.sme.randomize_smiles(sm)  # Random transoform
        if sm_r is None:
            sm_spaced = split(sm)  # Spacing
        else:
            sm_spaced = split(sm_r)  # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split) <= MAX_LEN - 2:
            return sm_split  # List
        else:
            return split(sm).split()

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)


class SmilesDatasetP1(Dataset):

    def __init__(self, file, vocab, seq_len=220, transform=Randomizer()):
        self.smiles = pd.read_csv(file)['smiles']
        self.output = pd.read_csv(file)['P1']
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        output = self.output[item].astype(float)
        sm = self.transform(sm)  # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)

        return torch.FloatTensor(X), output


class SmilesDatasetP1_P9(Dataset):

    def __init__(self, file, vocab, seq_len=220, transform=Randomizer()):
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
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        output1 = self.output1[item].astype(float)
        output2 = self.output2[item].astype(float)
        output3 = self.output3[item].astype(float)
        output4 = self.output4[item].astype(float)
        output5 = self.output5[item].astype(float)
        output6 = self.output6[item].astype(float)
        output7 = self.output7[item].astype(float)
        output8 = self.output8[item].astype(float)
        output9 = self.output9[item].astype(float)
        sm = self.transform(sm)  # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)

        return torch.FloatTensor(X), torch.tensor(
            [[output1, output2, output3, output4, output5, output6, output7, output8, output9]])


class SmilesDataset_singleline():
    def __init__(self, input1, vocab, seq_len=220, transform=Randomizer()):
        self.input1 = input1
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def get_item(self):
        sm = self.transform(self.input1)  # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)

        return torch.FloatTensor(X)

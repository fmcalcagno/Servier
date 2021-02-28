import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM Model including an Embedding layer from the outputs of the sequences.
    The hidden output is preserved to increase performance.
    """
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=2, dropout=0.5,output_size=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(input_size, embed_size)
        self.LSTM = nn.LSTM(input_size=embed_size,hidden_size= hidden_size,dropout=dropout,num_layers= n_layers)

        self.out = nn.Linear(hidden_size * input_size, 256)
        self.relu = nn.ReLU()
        self.out2 = nn.Linear(256,  output_size)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.LSTM(embedded)
        output = self.out(outputs.view(len(src),-1))
        output = self.relu(output)
        output = self.out2(output)
        return output,hidden


class LinearClassificationX(nn.Module):
    """
    This is a second model I tried but the LSTM Model had better accuracy and F1 Score, I leave it for future use.
    """
    def __init__(self,input_dim=2048 ,hidden_dim=256,  tagset_size=1,dropout=0.1):
        super(LinearClassificationX, self).__init__()  # Number of input features is 12.
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, tagset_size)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, inputs):
        x= self.bn(inputs)
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


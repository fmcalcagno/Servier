from torch import nn


class LinearClassificationX(nn.Module):
    def __init__(self, input_dim,hidden_dim,  tagset_size,dropout):
        super(LinearClassificationX, self).__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim *2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_2 = nn.Linear(hidden_dim *2, hidden_dim )
        self.layer_final = nn.Linear(hidden_dim , tagset_size)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x=self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_final(x)
        return x
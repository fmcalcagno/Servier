from torch import nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class LinearClassificationX(nn.Module):
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


class LinearClassificationX2(nn.Module):
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
        x = torch.sigmoid(self.layer_final(x))
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5,tagset_size=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,bidirectional=True)

        self.layer_final = nn.Linear(hidden_size*input_size, tagset_size)

    def forward(self, src, hidden=None):
        # src: (T,B)

        embedded = self.embed(src)# (T,B,H)
        outputs, hidden = self.gru(embedded, hidden) # (T,B,2H), (2L,B,H)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])

        X = outputs.view(-1,outputs.shape[1]*outputs.shape[2])
        X = self.layer_final (X)

        return X ,hidden # (T,B,H), (2L,B,H)






class LSTMModel(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=2, dropout=0.5,output_size=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(input_size, embed_size)
        self.LSTM = nn.LSTM(input_size=embed_size,hidden_size= hidden_size,dropout=dropout,num_layers= n_layers)
        self.out = nn.Linear(hidden_size * input_size, output_size)

    def forward(self, src, hidden=None):
        # src: (T,B)

        embedded = self.embed(src)
        outputs, hidden = self.LSTM(embedded)
        output = self.out(outputs.view(len(src),-1))
        return output,hidden



class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        # src: (T,B)
        embedded = self.embed(src)# (T,B,H)
        outputs, hidden = self.gru(embedded, hidden) # (T,B,2H), (2L,B,H)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden # (T,B,H), (2L,B,H)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1) # log???
        return output, hidden, attn_weights



class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):  # (T,B)
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()  # (T,B,V)
        encoder_output, hidden = self.encoder(src)  # (T,B,H), (2L,B,H)
        hidden = hidden[:self.decoder.n_layers]  # (L,B,H)
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)  # (B,V), (L,B,H)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(dim=1)[1]  # (B)
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs  # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.encoder.embed(src)  # (T,B,H)
        _, hidden = self.encoder.gru(embedded, None)  # (T,B,2H), (2L,B,H)
        hidden = hidden.detach().numpy()
        return np.hstack(hidden[2:])  # (B,4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        else:  # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out
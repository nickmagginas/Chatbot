import torch
import read_files as r

### A GRU based decoder
class Decoder(torch.nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim = 2)

    def forward(self, x, s):
        embedded = torch.nn.functional.relu(self.embedding(x))
        reccurent_weights, hidden = self.gru(embedded, s)
        output = self.fc(reccurent_weights)
        return self.softmax(output), hidden

### Much more complex architechture utilizing Attention
class AttentionDecoder(torch.nn.Module):
    def __init__(self, output_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.attention = torch.nn.Linear(2 * hidden_size, r.MAX_LENGTH)
        self.gru = torch.nn.GRU(2 * hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.att_softmax = torch.nn.Softmax(dim = 2)
        self.softmax = torch.nn.LogSoftmax(dim = 2)

    def forward(self, x, s, encoder_outputs):
        (encoder_length, batch_size, _) = encoder_outputs.shape
        embedded = self.embedding(x)
        attention_input = torch.cat((embedded, s), dim = 2)
        attention_w = self.attention(attention_input)
        attention_coefficient = self.att_softmax(attention_w).view(r.MAX_LENGTH, batch_size, 1)
        extended_outputs = torch.zeros(r.MAX_LENGTH, batch_size, self.hidden_size)
        extended_outputs[:encoder_length] = encoder_outputs
        encoder_attention = torch.mul(attention_coefficient, extended_outputs)
        collapsed_attention = torch.sum(encoder_attention, dim = 0)
        decoder_input = torch.cat((collapsed_attention.repeat(1, 1, 1), embedded), dim = 2)
        _, decoder_hidden = self.gru(decoder_input, s)
        output = self.fc(decoder_hidden)
        return self.softmax(output), decoder_hidden

##############################################################################################
#### The attention decoder also takes as input all encoder outputs for the pair.           ###
#### It ways them according to its current input and state sums them and merges            ###
#### the result with the current encoder input better encaptulating long term dependencies ###
##############################################################################################

import torch

### Initial Decoder -- No Attention
class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim = 1)

    def forward(self, x, s):
        embedded = torch.nn.functional.relu(self.embedding(x))
        o, ns = self.gru(embedded, s)
        fco = self.fc(o[0])
        return self.softmax(fco), ns

def feed_forward(decoder, x, s, m_length, accumulator = []):
    if accumulator == []:
        nx, ns = decoder.forward(x, s)
        return feed_forward(decoder, nx, ns, m_length, [nx])
    if len(accumulator) >= m_length:
        return accumulator
    nx, ns = decoder.forward(torch.argmax(x).view(-1, 1), s)
    return feed_forward(decoder, nx, ns, m_length, [nx] + accumulator)

def feed_forward_test(decoder, x, s, accumulator = []):
    if accumulator == []:
        nx, ns = decoder.forward(x, s)
        return feed_forward_test(decoder, nx, ns, [torch.argmax(nx).item()])
    if torch.argmax(x).item() == 2: return accumulator
    nx, ns = decoder.forward(torch.argmax(x).view(-1, 1), s)
    return feed_forward_test(decoder, nx, ns, [torch.argmax(nx).item()] + accumulator)





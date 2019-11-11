import encoder as e
import decoder as d
import read_dialogues as reader
import torch

FILENAME = 'data/dialogues/AGREEMENT_BOT.txt'
HIDDEN_SIZE = 512

onehot = lambda w, l: torch.Tensor([1 if w == 1 else 0 for i in range(l)])

### One hot
def prepare_for_loss(sentence, length):
    return [onehot(w, length).view(-1, length) for w in sentence]

### Main Training Routine
def main(verbose = False):
    dictionary, pairs = reader.prepare_data(FILENAME)
    rev = reader.reverse_dictionary(dictionary)
    length = dictionary.__len__()
    encoder = e.Encoder(length, HIDDEN_SIZE)
    decoder = d.Decoder(HIDDEN_SIZE, length)
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), 0.001)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), 0.001)
    loss_function = torch.nn.NLLLoss()
    SOS = torch.LongTensor([1]).view(-1, 1)
    for index, pair in enumerate(pairs):
        seq_length = pair[1].index(2)
        state = encoder.init_state(HIDDEN_SIZE)
        propagate = e.feed_sentence(encoder, pair[0], state)
        acc = d.feed_forward(decoder, SOS, propagate , seq_length)
        predictions = torch.cat(acc)
        target = torch.LongTensor(pair[1][1::])
        loss = loss_function(predictions, target)
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()
        if verbose:
            print(f'Q: {reader.translate(pair[0], rev)}')
            print(f'A: {reader.translate(pair[1], rev)}')
            print(f'Encoder State Dimension: {propagate.shape}')
            print(f'Predictions Shape: {predictions.shape}')
            print(f'Target Shape: {target.shape}')

        print(f'Iteration: {index}, Loss: {loss.item()}')
        decoder_optimizer.step()
        encoder_optimizer.step()
    return encoder, decoder


if __name__ == '__main__': main()



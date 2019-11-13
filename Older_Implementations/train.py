import encoder as e
import decoder as d
import read_dialogues as reader
import torch
from time import sleep

FILENAME = 'data/dialogues/AGREEMENT_BOT.txt'
HIDDEN_SIZE = 256


### Main Training Routine
def main(verbose = False):
    dictionary = reader.recover_dictionary('dictionary')
    pairs = reader.encode_corpus(reader.create_pairs(FILENAME), dictionary)
    rev = reader.reverse_dictionary(dictionary)
    length = dictionary.__len__()
    encoder = e.Encoder(length, HIDDEN_SIZE)
    decoder = d.Decoder(HIDDEN_SIZE, length)
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), 0.0005)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), 0.0005)
    loss_function = torch.nn.NLLLoss()
    SOS = torch.LongTensor([1]).view(-1, 1)
    for index, pair in enumerate(pairs):
        if verbose:
            print(reader.translate(pair[0], rev))
            print(reader.translate(pair[1], rev))
        seq_length = pair[1].index(2)
        state = encoder.init_state(HIDDEN_SIZE)
        propagate = e.feed_sentence(encoder, pair[0], state)
        acc = d.feed_forward(decoder, SOS, propagate , seq_length)
        predictions = torch.cat(acc)
        translatable = torch.argmax(predictions, dim = 1).numpy()
        translated = reader.translate(translatable, rev)
        target = torch.LongTensor(pair[1][1::])
        loss = loss_function(predictions, target)
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()
        if verbose:
            print(f'Q: {reader.translate(pair[0][1::], rev)}')
            print(f'A: {reader.translate(pair[1], rev)}')
            print(f'Encoder State Dimension: {propagate.shape}')
            print(f'Predictions: {translated}')
            print(f'Target:  {reader.translate(target.numpy(), rev)}')

        print(f'Iteration: {index}, Loss: {loss.item()}')
        decoder_optimizer.step()
        encoder_optimizer.step()
        if verbose: sleep(1)
    torch.save(encoder.state_dict(), 'encoder')
    torch.save(decoder.state_dict(), 'decoder')
    return encoder, decoder


if __name__ == '__main__': main(verbose = False)



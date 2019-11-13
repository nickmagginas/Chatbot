import read_dialogues as r
import torch
import encoder as e
import decoder as d

HIDDEN_SIZE = 256

def forward(encoder, decoder, question):
    state = encoder.init_state(256)
    encoded_state = e.feed_sentence(encoder, question, state)
    SOS_TOKEN = torch.LongTensor([1]).view(-1, 1)
    output = d.feed_forward_test(decoder, SOS_TOKEN, encoded_state)
    return output


def converse_loop(encoder, decoder, dictionary):
    q = input('Your Go: \n')
    filtered_question = r.remove_punctuation(q).lower()
    encoded_question = r.encode_sentence(filtered_question, dictionary)
    print(encoded_question)
    output = forward(encoder, decoder, encoded_question)
    print(r.translate(output, r.reverse_dictionary(dictionary)))


dictionary = r.recover_dictionary('dictionary')
encoder = e.Encoder(dictionary.__len__(), HIDDEN_SIZE)
decoder = d.Decoder(HIDDEN_SIZE, dictionary.__len__())
encoder.load_state_dict(torch.load('encoder'))
decoder.load_state_dict(torch.load('decoder'))
converse_loop(encoder, decoder, dictionary)


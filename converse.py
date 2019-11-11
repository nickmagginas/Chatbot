import read_dialogues as r
import torch
import encoder as e
import decoder as d
import train

HIDDEN_SIZE = 512

def forward(encoder, decoder, question):
    state = encoder.init_state(512)
    encoded_state = e.feed_sentence(encoder, question, state)
    SOS_TOKEN = torch.LongTensor([1]).view(-1, 1)
    output = d.feed_forward_test(decoder, SOS_TOKEN, encoded_state)
    return output


def converse_loop(encoder, decoder, dictionary):
    encoder, decoder = train.main()
    q = input('Your Go: \n')
    filtered_question = r.remove_punctuation(q).lower()
    encoded_question = r.encode_sentence(filtered_question, dictionary)
    print(encoded_question)
    output = forward(encoder, decoder, encoded_question)
    print(r.translate(output, r.reverse_dictionary(dictionary)))


dictionary = r.recover_dictionary('dictionary')

converse_loop(1, 2, dictionary)


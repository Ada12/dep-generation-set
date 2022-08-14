import torch

from module.seq2seq import Attention, EncoderRNN, LuongAttnDecoderRNN, Seq2Seq


def test_EncoderRNN():
    seq_len = 4
    input_size = 10
    embed_size = 8
    hidden_size = 6
    n_layers = 2
    batch_size = 3

    source = torch.LongTensor([
        [1, 2, 3, 0],
        [1, 2, 3, 4],
        [3, 4, 0, 0]
    ])

    encoder = EncoderRNN(
        input_size,
        embed_size,
        hidden_size,
        n_layers,
        dropout_p=.5,
        padding_idx=0
    )
    output, hidden = encoder(source)
    assert output.size() == (batch_size, seq_len, hidden_size)
    assert hidden.size() == (n_layers * 2, batch_size, hidden_size)


def test_Attention():
    batch_size = 3
    hidden_size = 5
    seq_len = 6
    hidden = torch.rand(batch_size, hidden_size)
    encoder_output = torch.rand(batch_size, seq_len, hidden_size)

    attn = Attention(hidden_size)

    output = attn(hidden, encoder_output)
    assert output.size() == (batch_size, 1, seq_len)


def test_LuongAttnDecoderRNN():
    source_seq_len = 4
    target_seq_len = 5
    vocab_size = 10
    embed_size = 8
    hidden_size = 6
    n_layers = 2
    batch_size = 3

    encoder: EncoderRNN = EncoderRNN(
        vocab_size,
        embed_size,
        hidden_size,
        n_layers,
        dropout_p=.5,
        padding_idx=0
    )
    decoder = LuongAttnDecoderRNN(encoder.embedding, embed_size, hidden_size, vocab_size, n_layers, 0.5)
    seq2seq = Seq2Seq(encoder, decoder)


    source = torch.LongTensor([
        [1, 2, 3, 0],
        [1, 2, 3, 4],
        [3, 4, 0, 0]
    ])
    target = torch.LongTensor([
        [5, 1, 2, 3, 0],
        [5, 1, 2, 3, 4],
        [5, 3, 4, 0, 0]
    ])

    outputs = seq2seq(source, target)
    assert outputs.size() == (batch_size, target_seq_len, vocab_size)




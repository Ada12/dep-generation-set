import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DepPresentationLayer(nn.Module):
    def __init__(self,
                 vocab_size, embed_size, padding_idx,
                 dropout_p,
                 tags_linear_sizes, use_tags):
        super(DepPresentationLayer, self).__init__()
        self.use_tags = use_tags

        self.embed_seq = nn.Sequential(
            nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx),
            nn.Dropout(dropout_p)
        )

        if self.use_tags:
            tags_layers = []
            for i in range(1, len(tags_linear_sizes)):
                tags_layers.append(nn.Linear(tags_linear_sizes[i - 1], tags_linear_sizes[i]))
                tags_layers.append(nn.Dropout(dropout_p))
                tags_layers.append(nn.ReLU())
            self.tags_seq = nn.Sequential(*tags_layers)

    def forward(self, dep_ids, tags):
        """
        :param dep_ids: (batch_size, seq_len)
        :param tags: (batch_size, seq_len, tags_num)
        :return: (batch_size, seq_len, embed_size + tag_out_size)
        """
        out_embed = self.embed_seq(dep_ids)
        if not self.use_tags:
            return out_embed

        out_tags = self.tags_seq(tags)
        return torch.cat((out_embed, out_tags), dim=2)


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(input_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

        # Init v.
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        attn_energies = self.score(inputs)
        # (batch_size, seq_len) -> (batch_size, 1, seq_len)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, inputs):
        # (batch_size, seq_len, input_size) -> (batch_size, seq_len, hidden_size)
        energy = self.attn(inputs).tanh()
        # -> (batch_size, seq_len)
        return torch.sum(self.v * energy, dim=2)


class Deps2deps(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 tag_vocab_size, tags_linear_sizes, dropout_p, padding_idx,
                 attn_hidden_size,
                 hidden_linear_sizes,
                 use_tags):
        super(Deps2deps, self).__init__()

        # Dep presentation layer.
        self.dep_pre_layer = DepPresentationLayer(vocab_size, embed_size, padding_idx,
                                                  dropout_p,
                                                  [tag_vocab_size] + tags_linear_sizes, use_tags)
        if use_tags:
            pre_out_size = embed_size + tags_linear_sizes[-1]
        else:
            pre_out_size = embed_size

        # Attention layer.
        self.attn = Attention(pre_out_size, attn_hidden_size)

        # Hidden layers.
        hidden_linear_sizes = [pre_out_size] + hidden_linear_sizes + [vocab_size]
        hidden_layers = []
        for i in range(0, len(hidden_linear_sizes) - 2):
            hidden_layers.append(nn.Linear(hidden_linear_sizes[i], hidden_linear_sizes[i + 1]))
            hidden_layers.append(nn.Dropout(dropout_p))
            hidden_layers.append(nn.ReLU())
        self.hidden_seq = nn.Sequential(*hidden_layers)

        # Out layers.
        self.out_layer = nn.Linear(hidden_linear_sizes[-2], hidden_linear_sizes[-1])

    def forward(self, source_dep_ids, source_tags):
        """
        :param source_dep_ids: (batch_size, seq_len).
        :param source_tags: (batch_size, seq_len, tags_num).
        :return: (batch_size, vocab_size)
        """
        # -> (batch_size, seq_len, embed_size + tag_out_size)
        dep_pre = self.dep_pre_layer(source_dep_ids, source_tags)

        attn_weights = self.attn(dep_pre)
        # (batch_size, 1, seq_len), (batch_size, seq_len, embed_size + tag_out_size)
        #   -> (batch_size, 1, embed_size + tag_out_size)
        x = attn_weights.bmm(dep_pre)
        x = x.squeeze(1)

        x = self.hidden_seq(x)
        out = self.out_layer(x)
        return out

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from odegcn import ODEG

# Model structure
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        """
        vocab_size: number of words in the vocabulary of the model
        embed_dim: dimension of word embeddings
        word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        word_gru_num_layers: number of layers in word-level GRU
        sent_gru_num_layers: number of layers in sentence-level GRU
        word_att_dim: dimension of word-level attention layer
        sent_att_dim: dimension of sentence-level attention layer
        use_layer_norm: whether to use layer normalization
        dropout: dropout rate; 0 to not use dropout
        """
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        self.fc = nn.Linear(sent_gru_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, line_tensor, adj):

        line_lengths = []
        sent_lengths = []

        for line in line_tensor:
            code_line = []
            line_lengths.append(len(line))

        
        line_tensor = line_tensor.type(torch.LongTensor)
        line_lengths = torch.tensor(line_lengths).type(torch.LongTensor).cuda()
        
        line_embeddings, word_att_weights, sent_att_weights, sents = self.sent_attention(line_tensor, line_lengths, adj)

        scores = self.fc(line_embeddings)
        final_scrs = self.sig(scores)

        return final_scrs, word_att_weights, sent_att_weights, sents

class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers, word_att_dim, use_layer_norm, dropout)


        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, line_tensor, line_lengths, adj):

        sents, word_att_weights = self.word_attention(line_tensor, line_lengths)

        sents = self.dropout(sents)

        line_tensor = sents
        sent_att_weights = None



        return line_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        embeddings: embeddings to init with
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        """
        self.embeddings.weight.requires_grad = freeze

    def forward(self, sents, sent_lengths):

        input = self.embeddings(sents.cuda())
        # input = input.permute(1, 0, 2)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(input)
        sents = final_hidden_state[-1]
        att_weights = None
        return sents, att_weights

